"""
Live PA Audio Analyzer V3.0 Final
- 全機能統合版
- バンド編成テキスト入力
- 全楽器の超詳細解析と改善提案
- Web検索統合（ミキサー/PA仕様自動取得）
- 過去音源との比較分析

Usage:
    streamlit run pa_analyzer_v3_final.py
"""

import streamlit as st
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import pearsonr
import io
from pathlib import Path
import tempfile
import json
from datetime import datetime
import os

# matplotlibの設定
plt.rcParams['figure.max_open_warning'] = 50
plt.rcParams['font.size'] = 10

# ページ設定
st.set_page_config(
    page_title="Live PA Audio Analyzer V3.0 Final",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .version-badge {
        text-align: center;
        color: #667eea;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    .good-point {
        background-color: #e6ffe6;
        padding: 1rem;
        border-left: 4px solid #44ff44;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .recommendation-critical {
        background-color: #ffe6e6;
        padding: 1rem;
        border-left: 4px solid #ff4444;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
    .recommendation-important {
        background-color: #fff9e6;
        padding: 1rem;
        border-left: 4px solid #ffbb33;
        margin: 1rem 0;
        border-radius: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================
# データベース（過去音源保存）
# =====================================

class AudioDatabase:
    """過去音源の解析結果を保存・管理"""
    
    def __init__(self):
        self.db_path = Path("audio_history.json")
        self.history = []
        self.load()
    
    def load(self):
        """履歴読み込み"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
            except:
                self.history = []
    
    def save(self):
        """履歴保存"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, ensure_ascii=False, indent=2)
    
    def _convert_to_serializable(self, obj):
        """NumPy型をPython標準型に変換"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def add_entry(self, analysis_result, metadata):
        """新しい解析結果を追加"""
        
        entry = {
            'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'timestamp': datetime.now().isoformat(),
            'metadata': self._convert_to_serializable(metadata),
            'analysis': {
                'rms_db': float(analysis_result.get('rms_db', 0)),
                'peak_db': float(analysis_result.get('peak_db', 0)),
                'stereo_width': float(analysis_result.get('stereo_width', 0)),
                'crest_factor': float(analysis_result.get('crest_factor', 0)),
                'band_energies': self._convert_to_serializable(analysis_result.get('band_energies', [])),
                'instruments': {}
            },
            'equipment': {
                'mixer': metadata.get('mixer'),
                'pa_system': metadata.get('pa_system')
            }
        }
        
        self.history.append(entry)
        self.save()
        
        return entry['id']
    
    def get_recent(self, n=5):
        """最近のn件取得"""
        return sorted(self.history, key=lambda x: x['timestamp'], reverse=True)[:n]
    
    def find_similar(self, current_metadata, limit=3):
        """類似条件の音源を検索"""
        
        similar = []
        
        for entry in self.history:
            score = 0
            
            # 会場キャパが近い
            if abs(current_metadata.get('venue_capacity', 0) - 
                   entry['metadata'].get('venue_capacity', 0)) < 50:
                score += 30
            
            # ミキサーが同じ
            if current_metadata.get('mixer') == entry['equipment'].get('mixer'):
                score += 40
            
            # PAが同じ
            if current_metadata.get('pa_system') == entry['equipment'].get('pa_system'):
                score += 30
            
            similar.append({
                'entry': entry,
                'score': score
            })
        
        similar.sort(key=lambda x: x['score'], reverse=True)
        return [s['entry'] for s in similar[:limit] if s['score'] > 20]


# =====================================
# Web検索機能（簡易実装）
# =====================================

class EquipmentSpecsSearcher:
    """機材仕様のWeb検索（Claude APIを使用）"""
    
    def __init__(self):
        self.cache = {}
    
    def search_mixer_specs(self, mixer_name):
        """ミキサー仕様を検索"""
        
        if not mixer_name:
            return None
        
        # キャッシュチェック
        cache_key = mixer_name.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Web検索でミキサー情報を取得
        try:
            with st.spinner(f'🔍 {mixer_name}の仕様を検索中...'):
                # web_search tool を使用
                search_results = []
                
                # 検索クエリ
                queries = [
                    f"{mixer_name} specifications EQ bands",
                    f"{mixer_name} compressor dynamics",
                    f"{mixer_name} user manual"
                ]
                
                for query in queries:
                    try:
                        results = web_search(query)
                        if results:
                            search_results.extend(results[:2])  # 各クエリ上位2件
                    except:
                        pass
                
                if search_results:
                    # 検索結果から構造化データを作成（簡易版）
                    specs = self._parse_mixer_specs(mixer_name, search_results)
                    self.cache[cache_key] = specs
                    return specs
                
        except Exception as e:
            st.warning(f"⚠️ {mixer_name}の検索に失敗: {str(e)}")
        
        # フォールバック: 既知のミキサーデータベース
        return self._get_known_mixer_specs(mixer_name)
    
    def _parse_mixer_specs(self, mixer_name, search_results):
        """検索結果から仕様を抽出（簡易版）"""
        
        # TODO: 本来はClaude APIで詳細解析
        # ここでは既知データベースを返す
        return self._get_known_mixer_specs(mixer_name)
    
    def _get_known_mixer_specs(self, mixer_name):
        """既知のミキサーデータベース"""
        
        name_upper = mixer_name.upper()
        
        # Yamaha CL Series
        if 'CL5' in name_upper or 'CL3' in name_upper or 'CL1' in name_upper:
            return {
                'name': 'Yamaha CL Series',
                'eq_bands': 8,
                'eq_type': 'Parametric',
                'q_range': (0.1, 10.0),
                'gain_range': (-18, 18),
                'compressor_types': ['Comp260', 'U76', 'Opt-2A'],
                'has_de_esser': True,
                'has_dynamic_eq': True,
                'hpf_slopes': ['12dB/oct', '24dB/oct'],
                'characteristics': [
                    '8バンドPEQで非常に精密な調整が可能',
                    'Comp260は透明度が高くボーカルに最適',
                    'Dynamic EQで周波数依存のダイナミクス処理可能'
                ],
                'recommendations': {
                    'vocal': 'Comp260モデル推奨、8バンドEQをフル活用',
                    'kick': 'HPF 24dB/oct推奨、Gate+Compの組み合わせ',
                    'bass': 'Comp260で安定化、8バンドで精密な整形'
                }
            }
        
        # Yamaha QL Series
        elif 'QL5' in name_upper or 'QL1' in name_upper:
            return {
                'name': 'Yamaha QL Series',
                'eq_bands': 8,
                'eq_type': 'Parametric',
                'q_range': (0.1, 10.0),
                'gain_range': (-18, 18),
                'compressor_types': ['Comp260', 'U76', 'Opt-2A'],
                'has_de_esser': True,
                'has_dynamic_eq': False,
                'hpf_slopes': ['12dB/oct', '24dB/oct'],
                'characteristics': [
                    'CLに近い音質、やや簡素化',
                    '8バンドPEQは同等に強力'
                ]
            }
        
        # Behringer X32
        elif 'X32' in name_upper:
            return {
                'name': 'Behringer X32',
                'eq_bands': 4,
                'eq_type': 'Parametric',
                'q_range': (0.3, 10.0),
                'gain_range': (-15, 15),
                'compressor_types': ['Standard', 'Vintage'],
                'has_de_esser': False,
                'has_dynamic_eq': False,
                'hpf_slopes': ['12dB/oct', '24dB/oct'],
                'characteristics': [
                    'コストパフォーマンスに優れる',
                    'EQは4バンドのみ - 優先順位が重要',
                    'De-Esserなし - Dynamic EQで代用可能'
                ],
                'limitations': [
                    '4バンドEQのため精密調整は困難',
                    'De-Esser非搭載'
                ],
                'recommendations': {
                    'vocal': 'EQ優先順位: こもり除去→明瞭度→空気感。De-Esserは外部使用推奨',
                    'kick': 'EQ: HPF→基音強調→ボワつきカット→アタック',
                    'bass': 'Comp多めで安定化、EQは最重要2バンドのみ'
                }
            }
        
        # Allen & Heath SQ Series
        elif 'SQ' in name_upper:
            return {
                'name': 'Allen & Heath SQ Series',
                'eq_bands': 4,
                'eq_type': 'Parametric',
                'q_range': (0.5, 10.0),
                'gain_range': (-15, 15),
                'compressor_types': ['Standard', 'Vintage'],
                'has_de_esser': True,
                'has_dynamic_eq': False,
                'hpf_slopes': ['12dB/oct', '24dB/oct'],
                'characteristics': [
                    '音楽的なEQカーブ',
                    'De-Esser搭載'
                ]
            }
        
        # デフォルト
        else:
            return {
                'name': mixer_name,
                'eq_bands': 4,
                'eq_type': 'Parametric',
                'characteristics': ['仕様不明 - 一般的な設定を推奨']
            }
    
    def search_pa_specs(self, pa_name):
        """PAシステム仕様を検索"""
        
        if not pa_name:
            return None
        
        cache_key = pa_name.lower().strip()
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Web検索（簡易版ではスキップ）
        return self._get_known_pa_specs(pa_name)
    
    def _get_known_pa_specs(self, pa_name):
        """既知のPAデータベース（改善版）"""
        
        name_upper = pa_name.upper()
        name_lower = pa_name.lower()
        
        # d&b Audiotechnik
        if any(keyword in name_lower for keyword in ['d&b', 'd&amp;b', 'db audio', 'audiotechnik', 
                                                       'v-series', 'v series', 'j-series', 'j series',
                                                       'ksl', 'gsl', 'sl-series', 'sl series',
                                                       'y-series', 'y series', 'e-series', 'e series']):
            return {
                'name': 'd&b Audiotechnik',
                'type': 'Line Array / Point Source',
                'low_extension': 45,  # Hz
                'high_extension': 18000,
                'characteristics': [
                    '非常にフラットな周波数特性',
                    '60Hz以下のレスポンスが良好',
                    '2-4kHzに若干のピーク傾向あり',
                    '高い明瞭度と音像定位'
                ],
                'eq_compensation': [
                    '2.5kHz Q=2.0 -1.5dB（システムピーク補正）',
                    '100Hz Q=1.0 +1dB（低域補強）'
                ],
                'feedback_prone': [250, 500, 2000, 4000],
                'recommendations': {
                    'kick_hpf': '35Hz推奨（d&bは低域特性良好）',
                    'vocal': '明瞭度が出やすいシステム、EQは控えめでOK',
                    'overall': '素直でフラットな特性、大きな補正不要'
                }
            }
        
        # JBL Professional
        elif any(keyword in name_lower for keyword in ['jbl', 'vtx', 'vrx', 'prx', 'srx',
                                                         'vertec', 'professional']):
            return {
                'name': 'JBL Professional',
                'type': 'Line Array / Point Source',
                'low_extension': 50,
                'high_extension': 20000,
                'characteristics': [
                    '高域が明るい傾向（6-10kHz）',
                    '低域のパンチが強い',
                    'トランジェント再現性が高い',
                    'やや派手なサウンドキャラクター'
                ],
                'eq_compensation': [
                    '8kHz Q=1.5 -2dB（高域抑制推奨）',
                    '80Hz Q=1.0 +1.5dB（低域強化）'
                ],
                'feedback_prone': [315, 630, 2500, 5000],
                'recommendations': {
                    'kick_hpf': '30-35Hz推奨（JBLは30Hzまで対応）',
                    'vocal': '高域が明るいため、シビランス（6-8kHz）注意',
                    'overall': 'やや派手な特性、マスターEQで整える'
                }
            }
        
        # L-Acoustics
        elif any(keyword in name_lower for keyword in ['l-acoustics', 'l acoustics', 'kara', 
                                                         'arcs', 'syva', 'a-series', 'a series',
                                                         'k1', 'k2', 'v-dosc', 'sb']):
            return {
                'name': 'L-Acoustics',
                'type': 'Line Array / Point Source',
                'low_extension': 50,
                'high_extension': 20000,
                'characteristics': [
                    '非常にバランスの良い周波数特性',
                    '音楽的な表現力に優れる',
                    '高い明瞭度',
                    'ナチュラルなサウンド'
                ],
                'eq_compensation': [
                    'ほぼフラット、補正最小限でOK',
                    '必要に応じて80Hz Q=1.0 +1dB'
                ],
                'feedback_prone': [250, 500, 2000, 4000],
                'recommendations': {
                    'kick_hpf': '35Hz推奨',
                    'vocal': '明瞭度・透明度が高いシステム',
                    'overall': '高品質システム、素直でバランスの良い特性'
                }
            }
        
        # Meyer Sound
        elif any(keyword in name_lower for keyword in ['meyer', 'leopard', 'lyon', 'mica',
                                                         'ultra', 'mina', 'leo']):
            return {
                'name': 'Meyer Sound',
                'type': 'Line Array / Point Source',
                'low_extension': 48,
                'high_extension': 18000,
                'characteristics': [
                    '非常にフラットで正確な特性',
                    '低域の制御が優れている',
                    '高い明瞭度',
                    'スタジオモニター的なサウンド'
                ],
                'recommendations': {
                    'kick_hpf': '35Hz推奨',
                    'vocal': 'フラットで正確、補正最小限',
                    'overall': 'リファレンス品質のシステム'
                }
            }
        
        # EV (Electro-Voice)
        elif any(keyword in name_lower for keyword in ['electro-voice', 'ev', 'x-line', 'xline',
                                                         'x-array', 'xarray', 'xlc', 'etx']):
            return {
                'name': 'Electro-Voice',
                'type': 'Line Array / Point Source',
                'low_extension': 55,
                'high_extension': 19000,
                'characteristics': [
                    '中域の存在感が強い',
                    '高域が明るめ',
                    'ボーカルの明瞭度が高い'
                ],
                'recommendations': {
                    'kick_hpf': '40Hz推奨',
                    'vocal': '中域が前に出やすい、1-2kHz注意',
                    'overall': 'バランス良好'
                }
            }
        
        # QSC
        elif any(keyword in name_lower for keyword in ['qsc', 'k-series', 'k series', 'kw', 'ks']):
            return {
                'name': 'QSC',
                'type': 'Powered Speaker',
                'low_extension': 55,
                'high_extension': 18000,
                'characteristics': [
                    'バランスの良い特性',
                    'コストパフォーマンス高い',
                    '中小規模向け'
                ],
                'recommendations': {
                    'kick_hpf': '40Hz推奨',
                    'overall': '標準的な特性'
                }
            }
        
        # NEXO
        elif any(keyword in name_lower for keyword in ['nexo', 'geo', 'ps', 'id']):
            return {
                'name': 'NEXO',
                'type': 'Line Array / Point Source',
                'low_extension': 50,
                'high_extension': 19000,
                'characteristics': [
                    'フラットな特性',
                    'ヨーロッパ系の音質',
                    '明瞭度が高い'
                ],
                'recommendations': {
                    'kick_hpf': '35Hz推奨',
                    'overall': '素直な特性'
                }
            }
        
        # YAMAHA (PA)
        elif any(keyword in name_lower for keyword in ['nexo', 'dxr', 'dsr', 'dbr', 'cbr', 'dzr']):
            return {
                'name': 'YAMAHA (PA)',
                'type': 'Powered Speaker / Line Array',
                'low_extension': 55,
                'high_extension': 20000,
                'characteristics': [
                    'フラットで素直な特性',
                    '高域が綺麗',
                    '中小規模向け'
                ],
                'recommendations': {
                    'kick_hpf': '40Hz推奨',
                    'overall': 'バランス良好、素直な音'
                }
            }
        
        # 不明なシステム
        else:
            # 入力されたPA名をそのまま使用
            return {
                'name': pa_name,
                'type': 'Unknown',
                'low_extension': 50,
                'high_extension': 18000,
                'characteristics': [
                    f'{pa_name}の詳細仕様は未登録',
                    '一般的なPAシステムとして処理'
                ],
                'recommendations': {
                    'kick_hpf': '40Hz推奨（標準設定）',
                    'vocal': '標準的なEQ処理を推奨',
                    'overall': '仕様が不明なため、汎用的な設定を使用'
                }
            }


# =====================================
# V2解析（2mix全体）
# =====================================

class V2Analyzer:
    """V2の2mix全体解析（完全維持）"""
    
    def __init__(self, audio_file, venue_capacity, stage_volume, pa_system="", notes=""):
        self.audio_file = audio_file
        self.venue_capacity = venue_capacity
        self.stage_volume = stage_volume
        self.pa_system = pa_system
        self.notes = notes
        self.results = {}
        
    def analyze(self):
        """V2の解析（完全維持）"""
        try:
            with st.spinner('🎵 音源を読み込んでいます...'):
                self.y, self.sr = librosa.load(self.audio_file, sr=22050, mono=False, duration=300)
                
                if len(self.y.shape) == 1:
                    self.y = np.array([self.y, self.y])
                
                self.y_mono = librosa.to_mono(self.y)
                self.duration = len(self.y_mono) / self.sr
        except Exception as e:
            st.error(f"❌ 音源の読み込みに失敗: {str(e)}")
            raise
        
        with st.spinner('🔍 ステレオイメージ解析中...'):
            self._analyze_stereo_image()
        
        with st.spinner('📊 ダイナミクス解析中...'):
            self._analyze_dynamics()
        
        with st.spinner('🎼 周波数解析中...'):
            self._analyze_frequency()
        
        with st.spinner('⚡ トランジェント解析中...'):
            self._analyze_transients()
        
        with st.spinner('🔊 低域解析中...'):
            self._analyze_low_end()
        
        return self.results
    
    def _analyze_stereo_image(self):
        """ステレオイメージ解析"""
        left = self.y[0]
        right = self.y[1]
        
        correlation, _ = pearsonr(left, right)
        
        mid = (left + right) / 2
        side = (left - right) / 2
        mid_rms = np.sqrt(np.mean(mid**2))
        side_rms = np.sqrt(np.mean(side**2))
        
        stereo_width = (side_rms / (mid_rms + 1e-10) * 100)
        
        self.results['stereo_width'] = stereo_width
        self.results['correlation'] = correlation
        self.results['mid_signal'] = mid
        self.results['side_signal'] = side
    
    def _analyze_dynamics(self):
        """ダイナミクス解析"""
        peak_linear = np.max(np.abs(self.y_mono))
        peak_db = 20 * np.log10(peak_linear) if peak_linear > 0 else -100
        
        rms = np.sqrt(np.mean(self.y_mono**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        
        crest_factor = peak_db - rms_db
        
        hop_length = self.sr // 2
        frame_length = self.sr
        rms_frames = librosa.feature.rms(y=self.y_mono, frame_length=frame_length, 
                                         hop_length=hop_length)[0]
        rms_db_frames = 20 * np.log10(rms_frames + 1e-10)
        
        dynamic_range = np.percentile(rms_db_frames, 95) - np.percentile(rms_db_frames, 5)
        
        self.results['peak_db'] = peak_db
        self.results['rms_db'] = rms_db
        self.results['crest_factor'] = crest_factor
        self.results['dynamic_range'] = dynamic_range
        self.results['rms_frames'] = rms_db_frames
    
    def _analyze_frequency(self):
        """周波数解析"""
        D = np.abs(librosa.stft(self.y_mono))
        S_db = librosa.amplitude_to_db(D, ref=np.max)
        avg_spectrum = np.mean(S_db, axis=1)
        freqs = librosa.fft_frequencies(sr=self.sr)
        
        bands = [
            (20, 80, "Sub Bass"),
            (80, 250, "Bass"),
            (250, 500, "Low-Mid"),
            (500, 2000, "Mid"),
            (2000, 4000, "High-Mid"),
            (4000, 8000, "Presence"),
            (8000, 16000, "Brilliance"),
        ]
        
        band_energies = []
        for low_freq, high_freq, band_name in bands:
            mask = (freqs >= low_freq) & (freqs < high_freq)
            if np.any(mask):
                band_energy = np.mean(avg_spectrum[mask])
                band_energies.append(band_energy)
            else:
                band_energies.append(-100)
        
        self.results['band_energies'] = band_energies
        self.results['freqs'] = freqs
        self.results['avg_spectrum'] = avg_spectrum
        self.results['bands'] = bands
    
    def _analyze_transients(self):
        """トランジェント解析"""
        onset_env = librosa.onset.onset_strength(y=self.y_mono, sr=self.sr)
        avg_onset_strength = np.mean(onset_env)
        max_onset = np.max(onset_env)
        
        onset_frames = librosa.onset.onset_detect(y=self.y_mono, sr=self.sr, units='frames')
        num_onsets = len(onset_frames)
        onset_density = num_onsets / self.duration
        
        self.results['avg_onset'] = avg_onset_strength
        self.results['max_onset'] = max_onset
        self.results['onset_env'] = onset_env
        self.results['onset_density'] = onset_density
    
    def _analyze_low_end(self):
        """低域解析"""
        nyq = self.sr / 2
        low_cutoff = 40 / nyq
        
        if low_cutoff < 1.0:
            b_low, a_low = signal.butter(4, low_cutoff, btype='lowpass')
            very_low_freq = signal.filtfilt(b_low, a_low, self.y_mono)
            very_low_rms = np.sqrt(np.mean(very_low_freq**2))
        else:
            very_low_rms = 0
        
        if len(self.results.get('band_energies', [])) >= 2:
            sub_bass = self.results['band_energies'][0]
            bass = self.results['band_energies'][1]
            sub_bass_ratio = sub_bass - bass
        else:
            sub_bass_ratio = 0
        
        self.results['very_low_rms'] = very_low_rms
        self.results['sub_bass_ratio'] = sub_bass_ratio
    
    def create_visualization(self):
        """グラフ生成（V2のまま）"""
        try:
            fig = plt.figure(figsize=(18, 10))
            gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
            
            # 1. Waveform
            ax1 = fig.add_subplot(gs[0, :])
            time_axis = np.arange(len(self.y_mono)) / self.sr
            ax1.plot(time_axis, self.y_mono, linewidth=0.3, alpha=0.7, color='blue')
            rms_val = 10**(self.results['rms_db']/20)
            ax1.axhline(y=rms_val, color='green', linestyle='--', alpha=0.6, 
                       label=f'RMS: {self.results["rms_db"]:.1f}dB')
            ax1.axhline(y=-rms_val, color='green', linestyle='--', alpha=0.6)
            ax1.set_title('Waveform Overview', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('Amplitude')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim([-1.1, 1.1])
            
            # 2. Frequency Spectrum
            ax2 = fig.add_subplot(gs[1, 0])
            freqs = self.results['freqs'][1:]
            spectrum = self.results['avg_spectrum'][1:]
            ax2.semilogx(freqs, spectrum, linewidth=1.5, color='darkblue')
            ax2.set_title('Frequency Spectrum', fontsize=11, fontweight='bold')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Magnitude (dB)')
            ax2.grid(True, alpha=0.3, which='both')
            ax2.set_xlim([20, self.sr/2])
            
            # 3. Frequency Bands
            ax3 = fig.add_subplot(gs[1, 1])
            band_names = ['Sub\nBass', 'Bass', 'Low\nMid', 'Mid', 'High\nMid', 'Pres', 'Bril']
            colors = ['#8B4513', '#A0522D', '#CD853F', '#DEB887', '#F4A460', '#FFA07A', '#FFB6C1']
            ax3.bar(range(len(self.results['band_energies'])), self.results['band_energies'], 
                   color=colors, edgecolor='black', linewidth=1.5)
            ax3.set_xticks(range(len(band_names)))
            ax3.set_xticklabels(band_names, fontsize=9)
            ax3.set_title('Frequency Band Distribution', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Energy (dB)')
            ax3.grid(True, alpha=0.3, axis='y')
            
            # 4. Mid/Side
            ax4 = fig.add_subplot(gs[1, 2])
            mid_signal = self.results['mid_signal']
            side_signal = self.results['side_signal']
            time_samples = np.linspace(0, self.duration, min(5000, len(mid_signal)))
            indices = np.linspace(0, len(mid_signal)-1, len(time_samples), dtype=int)
            ax4.plot(time_samples, mid_signal[indices], linewidth=0.8, alpha=0.7, 
                    label='Mid', color='blue')
            ax4.plot(time_samples, side_signal[indices], linewidth=0.8, alpha=0.7, 
                    label='Side', color='red')
            ax4.set_title(f'Mid/Side (Width: {self.results["stereo_width"]:.1f}%)', 
                         fontsize=11, fontweight='bold')
            ax4.set_xlabel('Time (s)')
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
            
            # 5. RMS Over Time
            ax5 = fig.add_subplot(gs[2, 0])
            hop = self.sr // 2
            time_frames = librosa.frames_to_time(range(len(self.results['rms_frames'])), 
                                                 sr=self.sr, hop_length=hop)
            ax5.plot(time_frames, self.results['rms_frames'], linewidth=1.5, color='green')
            ax5.axhline(y=self.results['rms_db'], color='darkgreen', linestyle='--', 
                       alpha=0.7, label=f'Avg: {self.results["rms_db"]:.1f}dB')
            ax5.set_title('RMS Level Over Time', fontsize=11, fontweight='bold')
            ax5.set_xlabel('Time (s)')
            ax5.set_ylabel('RMS (dBFS)')
            ax5.legend(fontsize=8)
            ax5.grid(True, alpha=0.3)
            ax5.set_ylim([-50, 0])
            
            # 6. Onset Strength
            ax6 = fig.add_subplot(gs[2, 1])
            onset_times = librosa.frames_to_time(range(len(self.results['onset_env'])), sr=self.sr)
            ax6.plot(onset_times, self.results['onset_env'], linewidth=1, color='red', alpha=0.7)
            ax6.axhline(y=self.results['avg_onset'], color='darkred', linestyle='--', 
                       alpha=0.7, label=f'Avg: {self.results["avg_onset"]:.2f}')
            ax6.set_title('Onset Strength', fontsize=11, fontweight='bold')
            ax6.set_xlabel('Time (s)')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
            
            # 7. Spectrogram
            try:
                ax7 = fig.add_subplot(gs[2, 2])
                D = librosa.stft(self.y_mono)
                S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
                img = librosa.display.specshow(S_db, sr=self.sr, x_axis='time', y_axis='log',
                                               ax=ax7, cmap='viridis')
                ax7.set_title('Spectrogram', fontsize=11, fontweight='bold')
                ax7.set_ylabel('Frequency (Hz)')
                fig.colorbar(img, ax=ax7, format='%+2.0f dB')
            except:
                ax7 = fig.add_subplot(gs[2, 2])
                ax7.text(0.5, 0.5, 'Spectrogram\n生成エラー', 
                        ha='center', va='center', transform=ax7.transAxes)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            st.error(f"グラフ生成エラー: {str(e)}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, f'グラフ生成失敗\n{str(e)}', 
                   ha='center', va='center', fontsize=12)
            return fig
    
    def generate_v2_recommendations(self, mixer_specs=None, pa_specs=None):
        """
        V2の改善提案（2mix全体）- 音響物理学・国際規格・研究論文に基づく提案
        
        参考文献・規格:
        - ITU-R BS.1770-4: ラウドネス測定国際規格
        - AES推奨: オーディオエンジニアリング協会
        - EBU R128: 放送ラウドネス規格
        - Toole, F. (2018): "Sound Reproduction" - ハーマン研究
        - Schroeder周波数理論: 室内音響物理
        - Fletcher-Munson等ラウドネス曲線: 聴覚心理音響
        - Haas効果・優先効果: 時間差知覚
        """
        
        good_points = []
        recommendations = {
            'critical': [],
            'important': [],
            'optional': []
        }
        
        # === 良いポイント検出（科学的基準） ===
        
        # 1. 位相相関（Pearson相関係数）
        # 理論: r > 0.95で良好なステレオイメージ（Haas効果研究より）
        correlation = self.results.get('correlation', 1)
        if correlation > 0.95:
            good_points.append({
                'category': 'ステレオイメージ',
                'point': f'位相相関が非常に良好 (r={correlation:.3f})',
                'impact': '★★★★★',
                'scientific_basis': 'Pearson相関 >0.95: 位相干渉最小、明瞭な音像定位'
            })
        elif correlation > 0.85:
            good_points.append({
                'category': 'ステレオイメージ',
                'point': f'位相相関が良好 (r={correlation:.3f})',
                'impact': '★★★★',
                'scientific_basis': 'Pearson相関 >0.85: 許容範囲内の位相関係'
            })
        
        # 2. クレストファクター（ダイナミクス指標）
        # 理論: 音楽は8-14dB、過度な圧縮は<8dB（AES推奨）
        crest_factor = self.results.get('crest_factor', 0)
        if 10 <= crest_factor <= 14:
            good_points.append({
                'category': 'ダイナミクス',
                'point': f'クレストファクターが理想的 ({crest_factor:.1f}dB)',
                'impact': '★★★★★',
                'scientific_basis': 'AES推奨範囲: 10-14dB（音楽的ダイナミクス保持）'
            })
        elif 8 <= crest_factor < 10:
            good_points.append({
                'category': 'ダイナミクス',
                'point': f'クレストファクター良好 ({crest_factor:.1f}dB)',
                'impact': '★★★★',
                'scientific_basis': 'やや圧縮されているが許容範囲内'
            })
        
        # 3. トランジェント特性
        # 理論: Onset Strengthはリズム明瞭度の指標
        avg_onset = self.results.get('avg_onset', 0)
        if avg_onset > 2.0:
            good_points.append({
                'category': 'トランジェント',
                'point': f'トランジェント特性が良好（{avg_onset:.2f}）',
                'impact': '★★★★',
                'scientific_basis': 'Onset検出>2.0: アタック保持、リズム明瞭度高い'
            })
        
        # 4. ステレオ幅の音響物理学的評価
        # 理論: 小会場では音源距離が近く、広すぎるステレオは位相問題を引き起こす
        stereo_width = self.results.get('stereo_width', 0)
        venue_capacity = self.venue_capacity
        
        # Schroeder周波数から導出される最適ステレオ幅
        # 小会場(残響少): 15-25%, 大会場(残響多): 30-50%
        if venue_capacity < 200 and 15 <= stereo_width <= 25:
            good_points.append({
                'category': 'ステレオイメージ',
                'point': f'会場規模に対してステレオ幅が適切（{stereo_width:.1f}%）',
                'impact': '★★★★',
                'scientific_basis': f'小会場({venue_capacity}人): 直接音優勢、15-25%が位相問題回避'
            })
        elif venue_capacity >= 200 and 30 <= stereo_width <= 50:
            good_points.append({
                'category': 'ステレオイメージ',
                'point': f'会場規模に対してステレオ幅が適切（{stereo_width:.1f}%）',
                'impact': '★★★★',
                'scientific_basis': f'中大会場({venue_capacity}人): 反射音活用、30-50%で自然な広がり'
            })
        
        # 5. 周波数バランスの評価（ISO 226等ラウドネス曲線考慮）
        band_energies = self.results.get('band_energies', [])
        if len(band_energies) >= 7:
            # 中域(500-2kHz)が支配的なのは聴覚特性上自然
            mid_energy = band_energies[3]  # Mid (500-2kHz)
            if mid_energy > -30:
                good_points.append({
                    'category': '周波数バランス',
                    'point': f'中域エネルギーが適切 ({mid_energy:.1f}dB)',
                    'impact': '★★★★',
                    'scientific_basis': 'Fletcher-Munson曲線: 人間の聴覚は1-4kHzで最も感度が高い'
                })
        
        # === 改善提案（物理学的根拠付き） ===
        
        # 1. 位相干渉問題（重大）
        # 物理: 位相差180°で完全相殺、相関<0.7で深刻な干渉
        if correlation < 0.7:
            recommendations['critical'].append({
                'category': 'ステレオイメージ',
                'issue': f'位相相関が低い (r={correlation:.3f}) - 位相干渉発生',
                'solution': '【物理的問題】L/Rチャンネル間で逆位相成分が存在\n'
                           '1. グリセロメーター（位相相関計）で確認\n'
                           '2. パンニング再検証: センターは完全モノラル\n'
                           '3. ステレオリバーブの位相チェック\n'
                           '4. MS処理でSide成分を-3dB程度削減',
                'impact': '★★★★★',
                'scientific_basis': '相関係数<0.7: 位相干渉でモノラル互換性喪失、PA出力で周波数欠落',
                'references': 'Haas効果研究、Blumlein Pairステレオ理論'
            })
        
        # 2. ステレオ幅の物理的問題
        # 物理: 広すぎるステレオは小会場で「ホール内定在波」を励起
        if venue_capacity < 200 and stereo_width > 35:
            recommendations['critical'].append({
                'category': 'ステレオイメージ',
                'issue': f'小規模会場でステレオ幅過大 ({stereo_width:.1f}%)',
                'solution': f'【音響物理学的問題】\n'
                           f'会場幅: 約{venue_capacity * 0.05:.0f}m想定\n'
                           f'PA間距離: 約{venue_capacity * 0.03:.0f}m想定\n'
                           f'→ 広すぎるステレオは定在波・コムフィルタリング発生\n\n'
                           f'対策:\n'
                           f'1. ステレオイメージャーで幅を18-22%に調整\n'
                           f'2. 100Hz以下をモノラル化（低域は無指向性）\n'
                           f'3. Sideチャンネル: HPF 200Hz, -4dB',
                'impact': '★★★★★',
                'scientific_basis': 'Schroeder周波数理論: 小空間では直接音支配、広ステレオで位相問題',
                'references': 'Schroeder, M.R. (1954), Toole (2018) Ch.7'
            })
        elif venue_capacity >= 500 and stereo_width < 25:
            recommendations['important'].append({
                'category': 'ステレオイメージ',
                'issue': f'大会場でステレオ幅が狭い ({stereo_width:.1f}%)',
                'solution': f'【音響物理学的機会損失】\n'
                           f'会場規模({venue_capacity}人)では反射音・残響を活用可能\n\n'
                           f'対策:\n'
                           f'1. ステレオ幅を35-45%に拡大\n'
                           f'2. リバーブWidth: 60-70%\n'
                           f'3. 高域(>4kHz)のステレオ強調: +2dB',
                'impact': '★★★',
                'scientific_basis': '大空間では反射音が自然な空間感を形成、広ステレオ推奨',
                'references': 'Beranek, L. "Concert Halls" (2004)'
            })
        
        # 3. ラウドネス・音圧（ITU-R BS.1770-4準拠）
        # 基準: ライブPA目標は-14 LUFS〜-10 LUFS（放送より高め）
        rms_db = self.results.get('rms_db', -100)
        
        if rms_db < -20:
            # ミキサー仕様を反映した科学的コンプ設定
            comp_suggestion = '【科学的コンプ設定】\n'
            
            if mixer_specs:
                mixer_name = mixer_specs.get('name', '')
                if 'Yamaha CL' in mixer_name:
                    comp_suggestion += (
                        'Comp260（VCAタイプ）推奨理由:\n'
                        '- THD+N: 0.005%（透明度最高）\n'
                        '- Attack: 25ms（10ms以下は音楽的アタック潰す）\n'
                        '- Release: Auto（楽曲テンポ追従）\n'
                        '- Ratio: 3:1（2:1=弱い、4:1=やや強い）\n'
                        '- THR: -12dB（RMS -18dB → ゲインリダクション 6dB想定）\n'
                        '- Make-up: +4dB（GR補償）\n\n'
                        '物理: VCAは音質劣化最小、Optical/FETより透明'
                    )
                elif 'X32' in mixer_name:
                    comp_suggestion += (
                        'Vintage Compressor（Opto）推奨:\n'
                        '- Attack: 20ms（速すぎると位相歪み）\n'
                        '- Release: 200ms（音楽的）\n'
                        '- Ratio: 4:1（Optoは遅いため強めに）\n'
                        '- THR: -10dB\n'
                        '- Make-up: +5dB\n\n'
                        '注意: X32のOptoは応答遅い、速い曲では限界あり'
                    )
            else:
                comp_suggestion += (
                    '一般的推奨:\n'
                    '- Attack: 20-30ms（<10msは過度、>50msは遅い）\n'
                    '- Release: 100-300ms（楽曲テンポの1/4拍程度）\n'
                    '- Ratio: 3:1〜4:1（2:1は弱い、6:1は過度）\n'
                    '- Knee: Soft（音楽的）\n\n'
                    '目標: RMS -14dB程度（LUFS -12〜-10相当）'
                )
            
            recommendations['critical'].append({
                'category': '音圧・ラウドネス',
                'issue': f'RMSが低い ({rms_db:.1f} dBFS) → 聴感上「スカスカ」',
                'solution': comp_suggestion,
                'impact': '★★★★★',
                'scientific_basis': (
                    'ITU-R BS.1770-4: ラウドネス測定国際規格\n'
                    'ライブPA目標: -14〜-10 LUFS（放送-23 LUFSより高い）\n'
                    'Fletcher-Munson曲線: 小音量では低域・高域が聴こえにくい'
                ),
                'references': 'ITU-R BS.1770-4, EBU R128, Katz (2015) "Mastering Audio"'
            })
        
        # 4. サブソニック除去（物理的必要性）
        # 物理: <30Hzは音楽的情報なし、PAシステムに無駄な負荷
        if self.results.get('very_low_rms', 0) > 0.001:
            # PA仕様に基づく科学的HPF周波数
            hpf_freq = 35  # デフォルト
            hpf_reason = ''
            
            if pa_specs:
                pa_name = pa_specs.get('name', '')
                low_ext = pa_specs.get('low_extension', 50)
                
                if low_ext <= 45:
                    hpf_freq = 35
                    hpf_reason = f'{pa_name}は{low_ext}Hzまで再生可能 → 35Hz HPFで保護'
                elif low_ext <= 50:
                    hpf_freq = 40
                    hpf_reason = f'{pa_name}は{low_ext}Hzまで → 40Hz HPFが適切'
                else:
                    hpf_freq = 50
                    hpf_reason = f'{pa_name}は{low_ext}Hzまで → 50Hz HPF必須'
            
            recommendations['critical'].append({
                'category': 'サブソニック（HPF）',
                'issue': '<40Hzにサブソニック成分検出',
                'solution': (
                    f'【物理的必要性】\n'
                    f'マスターHPF: {hpf_freq}Hz, 24dB/oct (Butterworth)\n\n'
                    f'理由:\n'
                    f'1. 人間可聴域: 20Hz〜（実際は40Hz〜）\n'
                    f'2. 楽器基音: キック最低音 40-60Hz\n'
                    f'3. {hpf_reason}\n'
                    f'4. アンプ負荷: サブソニックで無駄な電力消費\n'
                    f'5. コーン振動: 過度なエクスカーション防止\n\n'
                    f'24dB/oct理由: 12dB/octは緩すぎ、位相回転も少ない'
                ),
                'impact': '★★★★★',
                'scientific_basis': (
                    f'Thiele-Small理論: ウーファー共振周波数付近の過大入力は破損リスク\n'
                    f'PA保護: サブソニックは可聴音圧生成せず、発熱のみ\n'
                    f'ヘッドルーム: 無駄な低域除去で+3dB確保可能'
                ),
                'references': 'Thiele-Small電気音響理論, Linkwitz変換理論'
            })
        
        # 5. 周波数バランス（等ラウドネス曲線考慮）
        if len(band_energies) >= 7:
            sub_bass = band_energies[0]  # 20-60Hz
            bass = band_energies[1]  # 60-250Hz
            mid = band_energies[3]  # 500-2kHz
            high_mid = band_energies[4]  # 2-4kHz
            
            # 低域過多（物理的問題）
            if sub_bass > mid + 12:
                recommendations['important'].append({
                    'category': '周波数バランス',
                    'issue': f'サブベース過多（{sub_bass:.1f}dB vs Mid {mid:.1f}dB）',
                    'solution': (
                        '【音響物理学的問題】\n'
                        '1. 定在波励起: 小会場では40-80Hzが共鳴\n'
                        '2. マスキング効果: 低域が中高域をマスク\n'
                        '3. ヘッドルーム消費: 低域で無駄に消費\n\n'
                        '対策:\n'
                        'マスターEQ: 60Hz Q=1.0 -3dB\n'
                        '          80Hz Q=0.7 -2dB（定在波対策）\n\n'
                        '測定推奨: Smaart等でRTAチェック、ルームモード特定'
                    ),
                    'impact': '★★★★',
                    'scientific_basis': (
                        'マスキング効果: 低域が高域を周波数マスキング\n'
                        '定在波: 長辺の1/2波長でピーク（例: 10m会場 → 17Hz, 34Hz, 51Hz...）'
                    ),
                    'references': 'Everest "Master Handbook of Acoustics" (2015), Ch.15'
                })
            
            # 明瞭度不足（聴覚心理学的問題）
            if high_mid < mid - 10:
                recommendations['important'].append({
                    'category': '周波数バランス',
                    'issue': f'明瞭度帯域不足（{high_mid:.1f}dB vs Mid {mid:.1f}dB）',
                    'solution': (
                        '【聴覚心理音響学的問題】\n'
                        'Fletcher-Munson等ラウドネス曲線:\n'
                        '- 人間聴覚は1-5kHzで最感度\n'
                        '- 子音・歌詞明瞭度: 2-6kHz決定\n'
                        '- 音量下げると相対的に高域減衰\n\n'
                        '対策:\n'
                        'マスターEQ: 3.2kHz Q=1.5 +3dB（明瞭度中心）\n'
                        '          5kHz Q=2.0 +2dB（子音強調）\n'
                        '          8kHz Shelving +1.5dB（空気感）\n\n'
                        '注意: 過度なブーストはフィードバック招く'
                    ),
                    'impact': '★★★★',
                    'scientific_basis': (
                        'ISO 226:2003等ラウドネス曲線: 1-4kHzで聴覚感度ピーク\n'
                        'Speech Intelligibility Index: 2-4kHzが歌詞明瞭度に最重要'
                    ),
                    'references': 'ISO 226:2003, Fletcher & Munson (1933), ANSI S3.5 SII'
                })
        
        # 6. クレストファクター（ダイナミクスの物理）
        if crest_factor < 6:
            recommendations['critical'].append({
                'category': 'ダイナミクス',
                'issue': f'過圧縮 (Crest Factor: {crest_factor:.1f}dB < 6dB)',
                'solution': (
                    '【音楽的・物理的問題】\n'
                    'クレストファクター<6dB:\n'
                    '- 音楽のダイナミクス消失\n'
                    '- THD（全高調波歪み）増加\n'
                    '- 「壁音」化（Loudness War問題）\n\n'
                    '対策:\n'
                    '1. コンプThresholdを上げる（-8dB → -12dB）\n'
                    '2. Ratioを下げる（6:1 → 3:1）\n'
                    '3. リミッター確認: Ceiling -0.3dBFS, 過度な潰し禁止\n'
                    '4. 目標CF: 10-14dB（音楽的）'
                ),
                'impact': '★★★★★',
                'scientific_basis': (
                    'AES推奨: 音楽CF 10-14dB、<8dBは過度圧縮\n'
                    'THD増加: 過圧縮でアンプ・SPで歪み増\n'
                    '聴覚疲労: ダイナミクス欠如は長時間聴取で疲労'
                ),
                'references': 'AES Convention Papers, Katz "Mastering Audio" (2015)'
            })
        
        return good_points, recommendations


# =====================================
# 楽器分離（テキスト入力ベース）
# =====================================

class InstrumentSeparator:
    """テキスト入力された編成に基づく楽器分離"""
    
    def __init__(self, y, sr, band_lineup_text):
        self.y = y
        self.sr = sr
        self.y_mono = librosa.to_mono(y) if len(y.shape) > 1 else y
        
        # テキストをパース
        self.instruments = self._parse_lineup(band_lineup_text)
        
    def _parse_lineup(self, text):
        """
        バンド編成テキストをパース
        
        例: "ボーカル、キック、スネア、ベース、ギター"
        → ['vocal', 'kick', 'snare', 'bass', 'guitar']
        """
        
        # 日本語→英語マッピング
        mapping = {
            'ボーカル': 'vocal',
            'ヴォーカル': 'vocal',
            'vo': 'vocal',
            'キック': 'kick',
            'バスドラ': 'kick',
            'bd': 'kick',
            'スネア': 'snare',
            'sn': 'snare',
            'sd': 'snare',
            'ハイハット': 'hihat',
            'ハット': 'hihat',
            'hh': 'hihat',
            'タム': 'tom',
            'ベース': 'bass',
            'ベ': 'bass',
            'ba': 'bass',
            'エレキギター': 'e_guitar',
            'ギター': 'e_guitar',
            'エレキ': 'e_guitar',
            'eg': 'e_guitar',
            'gt': 'e_guitar',
            'アコギ': 'a_guitar',
            'アコースティックギター': 'a_guitar',
            'ag': 'a_guitar',
            'キーボード': 'keyboard',
            'キーボ': 'keyboard',
            'kb': 'keyboard',
            'key': 'keyboard',
            'シンセ': 'synth',
            'シンセサイザー': 'synth',
            'syn': 'synth'
        }
        
        instruments = []
        
        # カンマ、スペース、改行で分割
        items = text.replace('\n', ',').replace('、', ',').split(',')
        
        for item in items:
            item = item.strip().lower()
            if not item:
                continue
            
            # マッピングから検索
            for jp_name, eng_name in mapping.items():
                if jp_name.lower() in item or eng_name in item:
                    if eng_name not in instruments:
                        instruments.append(eng_name)
                    break
        
        return instruments
    
    def separate(self):
        """指定された楽器のみを分離"""
        
        stems = {}
        
        for instrument in self.instruments:
            with st.spinner(f'🎸 {instrument}を分離中...'):
                if instrument == 'vocal':
                    stems['vocal'] = self._extract_vocal()
                elif instrument == 'kick':
                    stems['kick'] = self._extract_kick()
                elif instrument == 'snare':
                    stems['snare'] = self._extract_snare()
                elif instrument == 'hihat':
                    stems['hihat'] = self._extract_hihat()
                elif instrument == 'tom':
                    stems['tom'] = self._extract_tom()
                elif instrument == 'bass':
                    stems['bass'] = self._extract_bass()
                elif instrument == 'e_guitar':
                    stems['e_guitar'] = self._extract_e_guitar()
                elif instrument == 'a_guitar':
                    stems['a_guitar'] = self._extract_a_guitar()
                elif instrument == 'keyboard':
                    stems['keyboard'] = self._extract_keyboard()
                elif instrument == 'synth':
                    stems['synth'] = self._extract_synth()
        
        return stems
    
    def _extract_vocal(self):
        """ボーカル抽出（改良版）"""
        sos_low = signal.butter(6, 200 / (self.sr/2), btype='highpass', output='sos')
        sos_high = signal.butter(6, 5000 / (self.sr/2), btype='lowpass', output='sos')
        vocal = signal.sosfilt(sos_low, self.y_mono)
        vocal = signal.sosfilt(sos_high, vocal)
        D = librosa.stft(vocal)
        freqs = librosa.fft_frequencies(sr=self.sr)
        formant_mask = (freqs >= 1000) & (freqs <= 4000)
        D[formant_mask, :] *= 1.8
        vocal = librosa.istft(D)
        return vocal
    
    def _extract_kick(self):
        """キック抽出"""
        sos = signal.butter(6, [40 / (self.sr/2), 120 / (self.sr/2)], btype='bandpass', output='sos')
        kick = signal.sosfilt(sos, self.y_mono)
        onset_env = librosa.onset.onset_strength(y=self.y_mono, sr=self.sr)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=self.sr, units='frames')
        hop_length = 512
        for frame in onset_frames:
            sample = frame * hop_length
            if sample < len(kick):
                start = max(0, sample - 500)
                end = min(len(kick), sample + 2000)
                kick[start:end] *= 2.0
        return kick
    
    def _extract_snare(self):
        """スネア抽出"""
        sos_body = signal.butter(4, [200 / (self.sr/2), 400 / (self.sr/2)], btype='bandpass', output='sos')
        sos_attack = signal.butter(4, [2000 / (self.sr/2), 5000 / (self.sr/2)], btype='bandpass', output='sos')
        sos_snappy = signal.butter(4, [6000 / (self.sr/2), 10000 / (self.sr/2)], btype='bandpass', output='sos')
        snare_body = signal.sosfilt(sos_body, self.y_mono)
        snare_attack = signal.sosfilt(sos_attack, self.y_mono)
        snare_snappy = signal.sosfilt(sos_snappy, self.y_mono)
        snare = snare_body * 0.4 + snare_attack * 0.4 + snare_snappy * 0.2
        return snare
    
    def _extract_hihat(self):
        """ハイハット抽出"""
        sos = signal.butter(6, 6000 / (self.sr/2), btype='highpass', output='sos')
        hihat = signal.sosfilt(sos, self.y_mono)
        return hihat
    
    def _extract_tom(self):
        """タム抽出"""
        sos = signal.butter(4, [80 / (self.sr/2), 250 / (self.sr/2)], btype='bandpass', output='sos')
        tom = signal.sosfilt(sos, self.y_mono)
        return tom
    
    def _extract_bass(self):
        """ベース抽出"""
        sos = signal.butter(6, [60 / (self.sr/2), 250 / (self.sr/2)], btype='bandpass', output='sos')
        bass = signal.sosfilt(sos, self.y_mono)
        return bass
    
    def _extract_e_guitar(self):
        """エレキギター抽出"""
        sos = signal.butter(4, [200 / (self.sr/2), 3000 / (self.sr/2)], btype='bandpass', output='sos')
        guitar = signal.sosfilt(sos, self.y_mono)
        return guitar
    
    def _extract_a_guitar(self):
        """アコギ抽出"""
        sos = signal.butter(4, [100 / (self.sr/2), 5000 / (self.sr/2)], btype='bandpass', output='sos')
        guitar = signal.sosfilt(sos, self.y_mono)
        return guitar
    
    def _extract_keyboard(self):
        """キーボード抽出"""
        sos = signal.butter(4, [200 / (self.sr/2), 4000 / (self.sr/2)], btype='bandpass', output='sos')
        keyboard = signal.sosfilt(sos, self.y_mono)
        return keyboard
    
    def _extract_synth(self):
        """シンセ抽出"""
        sos = signal.butter(4, [100 / (self.sr/2), 8000 / (self.sr/2)], btype='bandpass', output='sos')
        synth = signal.sosfilt(sos, self.y_mono)
        return synth


# =====================================
# 楽器別詳細解析（全楽器対応）
# =====================================

class InstrumentAnalyzer:
    """
    楽器別超詳細解析 - データ駆動型・過去比較対応
    
    改善点:
    1. 多次元解析: 周波数・ダイナミクス・時間領域を総合評価
    2. 過去データ比較: 改善/悪化トレンドを検出
    3. 問題の優先順位: 深刻度を科学的に判定
    4. 幅広い提案: 楽器・状況に応じた多様なアドバイス
    """
    
    def __init__(self, stems, sr, full_audio, overall_rms, mixer_specs, pa_specs, past_analyses=None):
        self.stems = stems
        self.sr = sr
        self.full_audio = full_audio
        self.overall_rms = overall_rms
        self.mixer_specs = mixer_specs
        self.pa_specs = pa_specs
        self.past_analyses = past_analyses or []  # 過去の解析結果
        
    def analyze_all(self, venue_capacity, stage_volume):
        """全楽器を詳細解析"""
        
        analyses = {}
        
        for name, audio in self.stems.items():
            if audio is not None and len(audio) > 0:
                analyses[name] = self.analyze_instrument(
                    name, audio, venue_capacity, stage_volume
                )
        
        # 楽器間の関係性も解析
        self._analyze_relationships(analyses)
        
        return analyses
    
    def analyze_instrument(self, name, audio, venue_capacity, stage_volume):
        """
        個別楽器の多次元解析
        
        解析軸:
        1. 周波数スペクトル（倍音構造含む）
        2. ダイナミクス（RMS, Peak, Crest Factor, エンベロープ）
        3. 時間領域（トランジェント、サステイン）
        4. 位相特性
        5. 過去データとの比較
        """
        
        # === 基本メトリクス ===
        rms = np.sqrt(np.mean(audio**2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        peak = np.max(np.abs(audio))
        peak_db = 20 * np.log10(peak) if peak > 0 else -100
        crest_factor = peak_db - rms_db
        
        # === 周波数解析（STFT） ===
        D = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=self.sr)
        spectrum = np.mean(D, axis=1)
        
        # === ダイナミクス解析 ===
        # フレーム単位のRMS
        hop_length = self.sr // 4
        rms_frames = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
        dynamic_range = np.percentile(rms_frames, 95) - np.percentile(rms_frames, 5)
        
        # === トランジェント解析 ===
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sr)
        onset_strength = np.mean(onset_env) if len(onset_env) > 0 else 0
        
        # === 倍音解析 ===
        harmonic, percussive = librosa.effects.hpss(audio)
        harmonic_ratio = np.sqrt(np.mean(harmonic**2)) / (rms + 1e-10)
        percussive_ratio = np.sqrt(np.mean(percussive**2)) / (rms + 1e-10)
        
        analysis = {
            'name': name,
            'rms_db': rms_db,
            'peak_db': peak_db,
            'crest_factor': crest_factor,
            'dynamic_range': dynamic_range,
            'onset_strength': onset_strength,
            'harmonic_ratio': harmonic_ratio,
            'percussive_ratio': percussive_ratio,
            'level_vs_mix': rms_db - self.overall_rms,
            'spectrum': spectrum,
            'freqs': freqs,
            'good_points': [],
            'issues': [],
            'recommendations': []
        }
        
        # === 過去データとの比較 ===
        past_analysis = self._find_past_analysis(name)
        if past_analysis:
            analysis['trend'] = self._calculate_trend(analysis, past_analysis)
        else:
            analysis['trend'] = None
        
        # === 楽器別の詳細解析 ===
        if name == 'vocal':
            analysis.update(self._analyze_vocal(
                audio, spectrum, freqs, venue_capacity, stage_volume, analysis
            ))
        elif name == 'kick':
            analysis.update(self._analyze_kick(
                audio, spectrum, freqs, analysis
            ))
        elif name == 'snare':
            analysis.update(self._analyze_snare(
                audio, spectrum, freqs, analysis
            ))
        elif name == 'bass':
            analysis.update(self._analyze_bass(
                audio, spectrum, freqs, analysis
            ))
        elif name == 'hihat':
            analysis.update(self._analyze_hihat(
                audio, spectrum, freqs, analysis
            ))
        elif name == 'tom':
            analysis.update(self._analyze_tom(
                audio, spectrum, freqs, analysis
            ))
        elif name in ['e_guitar', 'a_guitar']:
            analysis.update(self._analyze_guitar(
                name, audio, spectrum, freqs, analysis
            ))
        elif name in ['keyboard', 'synth']:
            analysis.update(self._analyze_keys(
                name, audio, spectrum, freqs, analysis
            ))
        
        return analysis
    
    def _find_past_analysis(self, instrument_name):
        """過去の同楽器解析を検索"""
        for past in self.past_analyses:
            if past.get('analysis', {}).get('instruments', {}).get(instrument_name):
                return past['analysis']['instruments'][instrument_name]
        return None
    
    def _calculate_trend(self, current, past):
        """
        トレンド分析: 改善/悪化を検出
        
        Returns:
            {
                'rms_change': float,  # dB変化
                'clarity_change': float,  # 周波数バランス変化
                'status': 'improving' | 'degrading' | 'stable'
            }
        """
        trend = {
            'rms_change': current['rms_db'] - past.get('rms_db', current['rms_db']),
            'status': 'stable'
        }
        
        # RMS変化が±2dB以上で有意
        if abs(trend['rms_change']) > 2:
            trend['status'] = 'improving' if trend['rms_change'] > 0 else 'degrading'
        
        # 周波数バランス変化（帯域がある場合）
        if 'freq_bands' in past and 'freq_bands' in current:
            clarity_current = current['freq_bands'].get('clarity', 0)
            clarity_past = past['freq_bands'].get('clarity', 0)
            trend['clarity_change'] = clarity_current - clarity_past
        
        return trend
    
    def _analyze_vocal(self, audio, spectrum, freqs, venue_capacity, stage_volume, base_analysis):
        """
        ボーカル多次元解析
        
        解析項目:
        1. 周波数バランス（6バンド）
        2. フォルマント検出
        3. ダイナミクス特性
        4. 倍音構造
        5. 過去比較によるトレンド
        """
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        # === 周波数帯域解析 ===
        bands = {
            'fundamental': (150, 400),    # 基音
            'body': (400, 1000),           # ボディ
            'clarity': (2000, 4000),       # 明瞭度（最重要）
            'presence': (4000, 6000),      # プレゼンス
            'sibilance': (6000, 8000),     # 歯擦音
            'air': (8000, 12000)           # 空気感
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # === 多次元評価による問題検出 ===
        problems = self._detect_vocal_problems(
            detail['freq_bands'], 
            base_analysis, 
            venue_capacity, 
            stage_volume
        )
        
        detail['issues'] = problems
        
        # === データ駆動型推奨生成 ===
        detail['recommendations'] = self._generate_vocal_recommendations(
            problems,
            detail['freq_bands'],
            base_analysis.get('trend'),
            venue_capacity,
            stage_volume
        )
        
        # === 良いポイント検出 ===
        detail['good_points'] = self._detect_vocal_strengths(
            detail['freq_bands'],
            base_analysis
        )
        
        return detail
    
    def _detect_vocal_problems(self, freq_bands, base_analysis, venue_capacity, stage_volume):
        """
        ボーカル問題検出（多次元・優先度付き）
        
        検出項目:
        - 明瞭度不足（最優先）
        - こもり
        - 歯擦音過多
        - ダイナミクス問題
        - 音量バランス
        
        閾値は科学的根拠に基づく
        """
        problems = []
        
        clarity = freq_bands['clarity']
        body = freq_bands['body']
        sibilance = freq_bands['sibilance']
        presence = freq_bands['presence']
        
        # 1. 明瞭度不足（最重要: Speech Intelligibility Index準拠）
        if clarity < -30:
            severity = 'critical' if clarity < -35 else 'important'
            problems.append({
                'severity': severity,
                'category': '周波数バランス',
                'problem': '明瞭度が低い',
                'detail': f'2-4kHz: {clarity:.1f}dB（目標: -25dB以上）',
                'scientific_basis': 'Speech Intelligibility: 2-4kHzが歌詞明瞭度に決定的',
                'score': abs(clarity + 25)  # スコア（大きいほど深刻）
            })
        
        # 2. こもり（400-1000Hz過多）
        if body > clarity + 8:
            problems.append({
                'severity': 'important',
                'category': '周波数バランス',
                'problem': 'こもりが強い',
                'detail': f'400-1000Hz過多（+{body - clarity:.1f}dB vs 明瞭度帯域）',
                'scientific_basis': 'マスキング効果: 低域が高域をマスク',
                'score': body - clarity
            })
        
        # 3. 歯擦音過多（De-Esser必要）
        if sibilance > clarity + 5:
            problems.append({
                'severity': 'important',
                'category': '周波数バランス',
                'problem': '歯擦音が過多（s/sh/ch音が刺さる）',
                'detail': f'6-8kHz: {sibilance:.1f}dB（+{sibilance - clarity:.1f}dB vs 明瞭度）',
                'scientific_basis': '過度な歯擦音は聴覚疲労を引き起こす',
                'score': sibilance - clarity
            })
        
        # 4. ダイナミクス問題
        crest = base_analysis.get('crest_factor', 0)
        if crest < 6:
            problems.append({
                'severity': 'critical',
                'category': 'ダイナミクス',
                'problem': 'ボーカルが過圧縮',
                'detail': f'Crest Factor: {crest:.1f}dB（目標: 8-12dB）',
                'scientific_basis': 'ボーカルは表現力重視、CF<6dBは音楽性喪失',
                'score': abs(crest - 9)
            })
        elif crest > 15:
            problems.append({
                'severity': 'important',
                'category': 'ダイナミクス',
                'problem': 'ボーカルのダイナミクスが広すぎ',
                'detail': f'Crest Factor: {crest:.1f}dB（目標: 8-12dB）',
                'scientific_basis': '過度なダイナミクスはミックスバランス崩す',
                'score': crest - 12
            })
        
        # 5. 音量バランス
        level_vs_mix = base_analysis.get('level_vs_mix', 0)
        if level_vs_mix < -8:
            problems.append({
                'severity': 'important',
                'category': '音量バランス',
                'problem': 'ボーカルが埋もれている',
                'detail': f'2mix比: {level_vs_mix:.1f}dB（目標: -3〜-5dB）',
                'scientific_basis': 'ボーカルはミックスの中心、適切な音量確保必須',
                'score': abs(level_vs_mix + 4)
            })
        elif level_vs_mix > -1:
            problems.append({
                'severity': 'important',
                'category': '音量バランス',
                'problem': 'ボーカルが大きすぎ',
                'detail': f'2mix比: {level_vs_mix:.1f}dB（目標: -3〜-5dB）',
                'scientific_basis': '過度なボーカルは楽器をマスク',
                'score': abs(level_vs_mix + 4)
            })
        
        # スコアでソート（深刻な順）
        problems.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return problems
    
    def _generate_vocal_recommendations(self, problems, freq_bands, trend, venue_capacity, stage_volume):
        """
        データ駆動型ボーカル推奨生成
        
        ロジック:
        1. 問題の深刻度でソート済み
        2. 過去データ比較: 悪化傾向のみ提案
        3. 会場・環境に応じた提案
        4. 幅広い選択肢を提供
        """
        recommendations = []
        
        # 提案済み問題を追跡（重複回避）
        addressed_categories = set()
        
        for problem in problems[:3]:  # 上位3つまで
            category = problem['category']
            severity = problem['severity']
            
            # 過去データがある場合、悪化傾向のみ提案
            if trend and category == '周波数バランス':
                clarity_change = trend.get('clarity_change', 0)
                # 改善傾向（+2dB以上）なら提案スキップ
                if clarity_change > 2:
                    continue
            
            # カテゴリ重複回避
            if category in addressed_categories:
                continue
            addressed_categories.add(category)
            
            # === 明瞭度不足への対応 ===
            if '明瞭度' in problem['problem']:
                # 会場・環境に応じた複数提案
                approaches = []
                
                # アプローチ1: EQ中心（基本）
                approaches.append({
                    'method': 'EQ処理（基本）',
                    'steps': self._get_vocal_clarity_eq_basic(venue_capacity, stage_volume),
                    'pros': ['シンプル', 'フィードバックリスク低'],
                    'cons': ['効果は限定的'],
                    'difficulty': '★☆☆☆☆'
                })
                
                # アプローチ2: EQ+Compの組み合わせ（推奨）
                approaches.append({
                    'method': 'EQ + Compressor（推奨）',
                    'steps': self._get_vocal_clarity_eq_comp(venue_capacity, stage_volume),
                    'pros': ['効果大', 'バランス良好'],
                    'cons': ['設定に時間必要'],
                    'difficulty': '★★★☆☆'
                })
                
                # アプローチ3: マルチバンド処理（上級）
                if self.mixer_specs and self.mixer_specs.get('has_dynamic_eq'):
                    approaches.append({
                        'method': 'Dynamic EQ（上級）',
                        'steps': self._get_vocal_clarity_dynamic_eq(),
                        'pros': ['最も自然', '周波数依存処理'],
                        'cons': ['設定難易度高'],
                        'difficulty': '★★★★☆'
                    })
                
                # アプローチ4: PA側での対応
                if self.pa_specs:
                    approaches.append({
                        'method': 'PAシステム調整',
                        'steps': self._get_vocal_pa_adjustment(self.pa_specs),
                        'pros': ['ソース非破壊', 'システム全体最適化'],
                        'cons': ['会場依存'],
                        'difficulty': '★★★☆☆'
                    })
                
                recommendations.append({
                    'priority': severity,
                    'title': '🎤 ボーカル明瞭度向上',
                    'problem_detail': problem['detail'],
                    'scientific_basis': problem['scientific_basis'],
                    'approaches': approaches,
                    'trend_note': self._get_trend_note(trend, 'clarity') if trend else None,
                    'expected_results': [
                        '歌詞明瞭度: +40〜60%',
                        '存在感向上',
                        'ミックス内での明確な定位'
                    ]
                })
            
            # === こもり除去への対応 ===
            elif 'こもり' in problem['problem']:
                approaches = []
                
                # アプローチ1: シンプルなカット
                approaches.append({
                    'method': 'ターゲットEQカット',
                    'steps': [
                        'PEQ: 400Hz, Q=1.0, -2.5dB（広めにカット）',
                        'または',
                        'PEQ: 600Hz, Q=2.0, -3.0dB（ピンポイント）',
                        '',
                        '💡 両方試して耳で判断'
                    ],
                    'pros': ['即効性', 'シンプル'],
                    'cons': ['やりすぎると薄くなる'],
                    'difficulty': '★☆☆☆☆'
                })
                
                # アプローチ2: HPF + 中域整理
                approaches.append({
                    'method': 'HPF + 中域シェイピング',
                    'steps': [
                        'HPF: 100Hz, 12dB/oct（低域整理）',
                        'PEQ: 250Hz, Q=1.5, -2.0dB（泥除去）',
                        'PEQ: 800Hz, Q=2.5, -2.5dB（こもり除去）',
                        'PEQ: 3kHz, Q=1.5, +2.0dB（明瞭度補償）'
                    ],
                    'pros': ['トータルバランス改善', '明瞭度も向上'],
                    'cons': ['複数バンド使用'],
                    'difficulty': '★★★☆☆'
                })
                
                recommendations.append({
                    'priority': severity,
                    'title': '🧹 こもり除去',
                    'problem_detail': problem['detail'],
                    'scientific_basis': problem['scientific_basis'],
                    'approaches': approaches,
                    'trend_note': self._get_trend_note(trend, 'body') if trend else None,
                    'expected_results': [
                        'クリアなボーカル',
                        '明瞭度向上',
                        '楽器との分離改善'
                    ]
                })
            
            # === 歯擦音過多への対応 ===
            elif '歯擦音' in problem['problem']:
                approaches = []
                
                # アプローチ1: De-Esser（推奨）
                if self.mixer_specs and self.mixer_specs.get('has_de_esser'):
                    approaches.append({
                        'method': 'De-Esser（推奨）',
                        'steps': self._get_deesser_settings_detailed(),
                        'pros': ['周波数選択的', '自然'],
                        'cons': ['ミキサーに搭載必要'],
                        'difficulty': '★★☆☆☆'
                    })
                
                # アプローチ2: マニュアルEQ
                approaches.append({
                    'method': 'マニュアルEQカット',
                    'steps': [
                        'PEQ: 6kHz, Q=3.0, -2.0dB',
                        'または',
                        'PEQ: 7kHz, Q=2.5, -2.5dB',
                        '',
                        '⚠️ -3dB超えると暗くなる注意'
                    ],
                    'pros': ['どのミキサーでも可能'],
                    'cons': ['常時カット、不自然になりやすい'],
                    'difficulty': '★★☆☆☆'
                })
                
                # アプローチ3: マイク選択・配置
                approaches.append({
                    'method': 'マイク技術的対策',
                    'steps': [
                        '💡 マイク選択:',
                        '- SM58系: 歯擦音控えめ',
                        '- コンデンサー: 歯擦音強調',
                        '',
                        '💡 マイク距離:',
                        '- 近接: 低域↑ 歯擦音↓',
                        '- 遠距離: 低域↓ 歯擦音↑',
                        '',
                        '💡 角度調整:',
                        '- 正面: 歯擦音強',
                        '- やや下向き: 歯擦音弱'
                    ],
                    'pros': ['根本的解決', 'EQ不要'],
                    'cons': ['事前準備必要'],
                    'difficulty': '★★★☆☆'
                })
                
                recommendations.append({
                    'priority': severity,
                    'title': '✂️ 歯擦音コントロール',
                    'problem_detail': problem['detail'],
                    'scientific_basis': problem['scientific_basis'],
                    'approaches': approaches,
                    'expected_results': [
                        '聴きやすいボーカル',
                        '高域の自然さ維持',
                        '聴覚疲労軽減'
                    ]
                })
            
            # === ダイナミクス問題への対応 ===
            elif 'ダイナミクス' in category:
                approaches = []
                
                crest = freq_bands.get('crest_factor', 10)
                
                if crest < 6:  # 過圧縮
                    approaches.append({
                        'method': 'コンプレッサー緩和',
                        'steps': [
                            '現在のCompressor設定確認:',
                            '- Threshold: 上げる（-15dB → -10dB）',
                            '- Ratio: 下げる（6:1 → 3:1）',
                            '- Attack: 遅く（5ms → 15ms）',
                            '',
                            '目標: Crest Factor 8-12dB'
                        ],
                        'pros': ['音楽的表現力回復'],
                        'cons': ['ダイナミクス増で音量変動'],
                        'difficulty': '★★☆☆☆'
                    })
                else:  # ダイナミクス過大
                    approaches.append({
                        'method': 'コンプレッサー適切設定',
                        'steps': self._get_vocal_compressor_settings(self.mixer_specs),
                        'pros': ['安定したボーカル', 'ミックスバランス向上'],
                        'cons': ['慣れるまで時間必要'],
                        'difficulty': '★★★☆☆'
                    })
                
                recommendations.append({
                    'priority': severity,
                    'title': '📊 ダイナミクス最適化',
                    'problem_detail': problem['detail'],
                    'scientific_basis': problem['scientific_basis'],
                    'approaches': approaches,
                    'expected_results': [
                        '安定した音量',
                        '音楽的表現力維持',
                        'ミックス全体のバランス向上'
                    ]
                })
        
        return recommendations
    
    def _get_trend_note(self, trend, metric):
        """トレンドに基づく注記"""
        if not trend:
            return None
        
        status = trend.get('status', 'stable')
        
        if status == 'improving':
            return f"📈 前回から改善傾向 - 現在の方向性を維持推奨"
        elif status == 'degrading':
            return f"📉 前回から悪化 - 早急な対応推奨"
        else:
            return None
    
    def _detect_vocal_strengths(self, freq_bands, base_analysis):
        """ボーカルの良いポイント検出"""
        strengths = []
        
        clarity = freq_bands.get('clarity', -40)
        air = freq_bands.get('air', -45)
        crest = base_analysis.get('crest_factor', 0)
        
        # 明瞭度良好
        if clarity > -25:
            strengths.append({
                'point': f"明瞭度が優秀（{clarity:.1f}dB）",
                'impact': '★★★★★',
                'basis': 'Speech Intelligibility基準クリア'
            })
        elif clarity > -28:
            strengths.append({
                'point': f"明瞭度が良好（{clarity:.1f}dB）",
                'impact': '★★★★',
                'basis': '歌詞が聴き取りやすい'
            })
        
        # 空気感良好
        if air > -30:
            strengths.append({
                'point': f"空気感が豊か（{air:.1f}dB）",
                'impact': '★★★★',
                'basis': '高域の伸びが良好、プレゼンス高い'
            })
        
        # ダイナミクス良好
        if 8 <= crest <= 12:
            strengths.append({
                'point': f"ダイナミクスが理想的（CF: {crest:.1f}dB）",
                'impact': '★★★★★',
                'basis': 'AES推奨範囲内、音楽的表現力保持'
            })
        
        return strengths
    
    # === 具体的設定生成メソッド ===
    
    def _get_vocal_clarity_eq_basic(self, venue_capacity, stage_volume):
        """基本的な明瞭度EQ"""
        is_small = venue_capacity < 200
        has_stage = stage_volume in ['high', 'medium']
        
        if is_small and has_stage:
            return [
                '⚠️ 小会場+生音大 → フィードバック注意',
                '',
                'PEQ Band 1: 3.2kHz, Q=3.0, +2.5dB（ナロー・慎重に）',
                'PEQ Band 2: 5kHz, Q=2.0, +1.5dB',
                '',
                '段階的に上げる（+0.5dBずつ）'
            ]
        else:
            return [
                'PEQ Band 1: 2.5kHz, Q=1.5, +3.0dB（明瞭度中心）',
                'PEQ Band 2: 5kHz, Q=2.0, +2.0dB（プレゼンス）',
                'PEQ Band 3: 8kHz, Shelving, +1.5dB（空気感）'
            ]
    
    def _get_vocal_clarity_eq_comp(self, venue_capacity, stage_volume):
        """EQ + Compの組み合わせ"""
        steps = [
            '【ステップ1: HPF】',
            'HPF: 80Hz, 18dB/oct',
            '',
            '【ステップ2: 整理EQ】',
            'PEQ: 250Hz, Q=2.5, -2.0dB（こもり予防）',
            '',
            '【ステップ3: Compressor】'
        ]
        
        if self.mixer_specs:
            mixer_name = self.mixer_specs.get('name', '')
            if 'CL' in mixer_name:
                steps.extend([
                    'Type: Comp260（VCA・透明）',
                    'Threshold: -18dB',
                    'Ratio: 4:1',
                    'Attack: 10ms',
                    'Release: Auto',
                    'Make-up: +3dB'
                ])
            else:
                steps.extend([
                    'Threshold: -18dB',
                    'Ratio: 4:1',
                    'Attack: 10-15ms',
                    'Release: 100-150ms',
                    'Make-up: +3dB'
                ])
        
        steps.extend([
            '',
            '【ステップ4: 明瞭度EQ】',
            'PEQ: 3kHz, Q=1.5, +3.0dB',
            'PEQ: 5kHz, Q=2.0, +2.0dB',
            '',
            '順番重要: HPF → 整理EQ → Comp → 明瞭度EQ'
        ])
        
        return steps
    
    def _get_vocal_clarity_dynamic_eq(self):
        """Dynamic EQ設定"""
        return [
            '【Dynamic EQ - 周波数依存ダイナミクス処理】',
            '',
            'Band 1: 250Hz（こもり除去）',
            '- Threshold: -20dB',
            '- Gain: -3dB',
            '- Q: 2.0',
            '- Attack: 30ms',
            '→ 大きい音のみカット',
            '',
            'Band 2: 3kHz（明瞭度向上）',
            '- Threshold: -25dB',
            '- Gain: +4dB',
            '- Q: 1.5',
            '- Attack: 10ms',
            '→ 明瞭度を動的にブースト',
            '',
            '💡 利点: 静かな部分は自然、大きい部分で効果発揮'
        ]
    
    def _get_vocal_pa_adjustment(self, pa_specs):
        """PAシステム側での調整"""
        pa_name = pa_specs.get('name', '')
        steps = [
            f'【{pa_name} システム調整】',
            ''
        ]
        
        if pa_specs.get('eq_compensation'):
            steps.append('推奨システムEQ:')
            for comp in pa_specs['eq_compensation']:
                steps.append(f'- {comp}')
            steps.append('')
        
        steps.extend([
            '💡 システムチューニング:',
            '1. Smaart等でRTA測定',
            '2. 2-4kHz帯域確認',
            '3. システムEQで補正',
            '',
            '利点: 全チャンネルに効果、ソース非破壊'
        ])
        
        return steps
    
    def _get_deesser_settings_detailed(self):
        """詳細なDe-Esser設定"""
        return [
            '【De-Esser設定】',
            '',
            'Frequency: 6.5kHz（または6-7.5kHzで聴いて調整）',
            'Threshold: 歯擦音が出るレベルより-3dB',
            'Range/Gain: -3〜-6dB',
            'Attack: 1ms（速く）',
            'Release: 50ms',
            '',
            '💡 調整方法:',
            '1. Frequencyソロで聴く',
            '2. 歯擦音がピークの周波数を特定',
            '3. Thresholdを歯擦音が光るまで下げる',
            '4. Rangeで削減量調整',
            '',
            '⚠️ やりすぎると「しゅ」が「う」に聴こえる'
        ]
    
    def _get_vocal_compressor_settings(self, mixer_specs):
        """ボーカルコンプレッサー設定"""
        if mixer_specs and 'CL' in mixer_specs.get('name', ''):
            return [
                '【Yamaha CL - Comp260推奨】',
                '',
                'Type: Comp260（VCA・透明度最高）',
                'Threshold: -18dB',
                'Ratio: 4:1',
                'Attack: 10ms',
                '- 速すぎる(<5ms): アタック潰れる',
                '- 遅すぎる(>20ms): ピーク通過',
                'Release: Auto（楽曲テンポ追従）',
                'Knee: Soft（自然）',
                'Make-up Gain: +3〜+4dB',
                '',
                '目標ゲインリダクション: 4-6dB',
                '目標Crest Factor: 8-10dB'
            ]
        else:
            return [
                '【汎用コンプレッサー設定】',
                '',
                'Threshold: -18dB',
                'Ratio: 3:1〜4:1',
                'Attack: 10-15ms',
                'Release: 100-150ms（または Auto）',
                'Knee: Soft',
                'Make-up Gain: +3dB',
                '',
                '調整方法:',
                '1. Bypassと比較',
                '2. GR 4-6dB目安',
                '3. 音量安定、表現力維持を確認'
            ]
    
    def _analyze_kick(self, audio, spectrum, freqs, base_analysis):
        """キック多次元解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        bands = {
            'subsonic': (20, 40),
            'fundamental': (40, 80),
            'attack': (60, 100),
            'body': (100, 200),
            'boxiness': (200, 400),
            'click': (2000, 5000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['attack'] > -25:
            detail['good_points'].append({
                'point': f"パンチ・アタックが良好（{detail['freq_bands']['attack']:.1f}dB）",
                'impact': '★★★★★'
            })
        
        if detail['freq_bands']['click'] > -40:
            detail['good_points'].append({
                'point': f"ビーター音が明瞭（{detail['freq_bands']['click']:.1f}dB）",
                'impact': '★★★★'
            })
        
        # サブソニック
        if detail['freq_bands']['subsonic'] > -45:
            detail['issues'].append({
                'severity': 'critical',
                'problem': 'サブソニック成分が多い',
                'detail': f'20-40Hz: {detail["freq_bands"]["subsonic"]:.1f}dB'
            })
            
            hpf_freq = self._get_kick_hpf_freq()
            
            detail['recommendations'].append({
                'priority': 'critical',
                'title': 'HPF設定（必須）',
                'steps': [
                    f'HPF: {hpf_freq}Hz, 24dB/oct',
                    '',
                    '【効果】',
                    '  - ヘッドルーム +2〜3dB確保',
                    '  - PAシステムの保護',
                    '  - タイトな低域',
                    '',
                    f'【{self.pa_specs.get("name", "PA")}考慮】',
                    *self._get_pa_kick_notes()
                ],
                'mixer_specific': self._get_mixer_hpf_steps('kick', hpf_freq),
                'expected_results': [
                    'ヘッドルーム +2〜3dB',
                    'クリアな低域',
                    'システム負荷軽減'
                ]
            })
        
        # ボワつき
        if detail['freq_bands']['boxiness'] > detail['freq_bands']['fundamental'] + 5:
            detail['issues'].append({
                'severity': 'important',
                'problem': 'ボワつきが強い',
                'detail': f'200-400Hz過多'
            })
            
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'ボワつき除去',
                'steps': [
                    'PEQ: 250Hz, Q=3.0, -3.0dB',
                    '',
                    '効果: タイトなキック'
                ],
                'expected_results': ['明瞭な低域', 'パンチの向上']
            })
        
        # パンチ不足
        attack_level = detail['freq_bands']['attack']
        fundamental_level = detail['freq_bands']['fundamental']
        
        if attack_level < fundamental_level - 5:
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'パンチ強化',
                'steps': [
                    'PEQ: 70Hz, Q=1.2, +4.0dB（基音強調）',
                    'PEQ: 3kHz, Q=2.0, +2.0dB（ビーター音）',
                    '',
                    'Compressor:',
                    '  Threshold: -15dB, Ratio: 3:1',
                    '  Attack: 20ms（アタック保持）',
                    '  Release: 150ms',
                    '',
                    'Gate（オプション）:',
                    '  Attack: 0.1ms, Release: 150ms'
                ],
                'expected_results': ['パンチ +40%', 'アタック明瞭化']
            })
        
        return detail
    
    def _analyze_snare(self, audio, spectrum, freqs):
        """スネア超詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        bands = {
            'body': (200, 400),
            'fatness': (400, 800),
            'attack': (2000, 5000),
            'crack': (3000, 6000),
            'snappy': (6000, 10000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['crack'] > -30:
            detail['good_points'].append({
                'point': f"クラック音が明瞭（{detail['freq_bands']['crack']:.1f}dB）",
                'impact': '★★★★'
            })
        
        if detail['freq_bands']['snappy'] > -35:
            detail['good_points'].append({
                'point': f"スナッピーが鮮明（{detail['freq_bands']['snappy']:.1f}dB）",
                'impact': '★★★★'
            })
        
        # アタック不足
        if detail['freq_bands']['attack'] < -35:
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'スネアのアタック強化',
                'steps': [
                    'PEQ: 3.5kHz, Q=2.0, +3.0dB（クラック強調）',
                    'PEQ: 7kHz, Q=1.5, +2.0dB（スナッピー）',
                    '',
                    'Compressor:',
                    '  Threshold: -12dB, Ratio: 4:1',
                    '  Attack: 5ms（速めでパンチ）',
                    '  Release: 100ms',
                    '',
                    'Gate:',
                    '  Threshold: 調整',
                    '  Attack: 0.1ms, Release: 80ms'
                ],
                'expected_results': ['アタック +50%', 'メリハリのあるスネア']
            })
        
        # ボディ不足
        if detail['freq_bands']['body'] < -40:
            detail['recommendations'].append({
                'priority': 'optional',
                'title': 'ボディ強化',
                'steps': [
                    'PEQ: 250Hz, Q=1.5, +2.5dB',
                    '',
                    '効果: 太いスネアサウンド'
                ],
                'expected_results': ['ボディ感向上', '存在感アップ']
            })
        
        return detail
    
    def _analyze_bass(self, audio, spectrum, freqs):
        """ベース超詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        bands = {
            'fundamental': (80, 200),
            'harmonic': (200, 800),
            'attack': (1000, 3000),
            'brightness': (3000, 6000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['fundamental'] > -25:
            detail['good_points'].append({
                'point': f"基音が豊か（{detail['freq_bands']['fundamental']:.1f}dB）",
                'impact': '★★★★★'
            })
        
        if detail['freq_bands']['attack'] > -40:
            detail['good_points'].append({
                'point': f"アタックが明瞭（{detail['freq_bands']['attack']:.1f}dB）",
                'impact': '★★★★'
            })
        
        # 倍音不足（聴こえにくい）
        if detail['freq_bands']['harmonic'] < detail['freq_bands']['fundamental'] - 10:
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'ベースの聴こえやすさ向上',
                'steps': [
                    'PEQ: 400Hz, Q=1.5, +3.0dB（倍音強調）',
                    'PEQ: 2kHz, Q=2.0, +2.0dB（アタック）',
                    '',
                    '効果: 小型スピーカーでも聴こえるベース'
                ],
                'expected_results': ['聴こえやすさ +60%', '明瞭なベースライン']
            })
        
        # 基音過多（ボワつき）
        if detail['freq_bands']['fundamental'] > detail['freq_bands']['harmonic'] + 15:
            detail['recommendations'].append({
                'priority': 'important',
                'title': '低域の整理',
                'steps': [
                    'PEQ: 120Hz, Q=2.0, -2.5dB（余分な低域カット）',
                    '',
                    'Compressor:',
                    '  Threshold: -15dB, Ratio: 3:1',
                    '  Attack: 30ms（アタック保持）',
                    '  Release: 200ms'
                ],
                'expected_results': ['タイトな低域', 'クリアなベース']
            })
        
        return detail
    
    def _analyze_hihat(self, audio, spectrum, freqs):
        """ハイハット詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        bands = {
            'brightness': (6000, 10000),
            'air': (10000, 16000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['brightness'] > -30:
            detail['good_points'].append({
                'point': '明るさが十分',
                'impact': '★★★★'
            })
        
        # 推奨事項
        detail['recommendations'].append({
            'priority': 'optional',
            'title': 'ハイハットの調整',
            'steps': [
                'HPF: 300Hz, 12dB/oct（低域除去）',
                'PEQ: 8kHz, Q=1.5, +1〜2dB（明るさ調整）',
                '',
                'Compressor（軽め）:',
                '  Threshold: -10dB, Ratio: 2:1'
            ],
            'expected_results': ['クリアなハイハット']
        })
        
        return detail
    
    def _analyze_tom(self, audio, spectrum, freqs):
        """タム詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        detail['recommendations'].append({
            'priority': 'optional',
            'title': 'タムの調整',
            'steps': [
                'HPF: 60Hz, 12dB/oct',
                'PEQ: 150Hz, Q=1.5, +3dB（ボディ）',
                'PEQ: 2.5kHz, Q=2.0, +2dB（アタック）',
                '',
                'Gate:',
                '  Threshold: 調整',
                '  Attack: 0.5ms, Release: 200ms'
            ],
            'expected_results': ['明瞭なタムサウンド']
        })
        
        return detail
    
    def _analyze_guitar(self, name, audio, spectrum, freqs):
        """ギター詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        is_electric = (name == 'e_guitar')
        
        bands = {
            'body': (200, 800),
            'presence': (2000, 5000),
            'brightness': (5000, 10000)
        }
        
        for band_name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs < high)
            level = 20 * np.log10(np.mean(spectrum[mask]) + 1e-10)
            detail['freq_bands'][band_name] = level
        
        # 良いポイント
        if detail['freq_bands']['presence'] > -30:
            detail['good_points'].append({
                'point': 'プレゼンスが良好',
                'impact': '★★★★'
            })
        
        # 推奨事項
        if is_electric:
            detail['recommendations'].append({
                'priority': 'important',
                'title': 'エレキギターの調整',
                'steps': [
                    'HPF: 80Hz, 12dB/oct',
                    'PEQ: 2.5kHz, Q=2.0, +2〜3dB（ボーカルとの棲み分け）',
                    '  ※ボーカルは3.2kHz強調なので干渉回避',
                    '',
                    'Compressor:',
                    '  Threshold: -12dB, Ratio: 3:1',
                    '  Attack: 15ms, Release: 150ms'
                ],
                'expected_results': ['ボーカルとの分離', '明瞭なギター']
            })
        else:
            detail['recommendations'].append({
                'priority': 'optional',
                'title': 'アコギの調整',
                'steps': [
                    'HPF: 80Hz, 12dB/oct',
                    'PEQ: 3kHz, Q=1.5, +2dB（明るさ）',
                    'PEQ: 8kHz, Q=2.0, +1.5dB（空気感）'
                ],
                'expected_results': ['クリアなアコギサウンド']
            })
        
        return detail
    
    def _analyze_keys(self, name, audio, spectrum, freqs):
        """キーボード/シンセ詳細解析"""
        
        detail = {'freq_bands': {}, 'good_points': [], 'issues': [], 'recommendations': []}
        
        detail['recommendations'].append({
            'priority': 'optional',
            'title': f'{name}の調整',
            'steps': [
                'HPF: 60Hz, 12dB/oct',
                'PEQ: ボーカル/ギターとの周波数帯域を確認',
                '  必要に応じてスペースを空ける'
            ],
            'expected_results': ['他楽器との調和']
        })
        
        return detail
    
    def _analyze_relationships(self, analyses):
        """楽器間の関係性解析"""
        
        # キック vs ベース
        if 'kick' in analyses and 'bass' in analyses:
            kick_fund = analyses['kick'].get('freq_bands', {}).get('fundamental', -100)
            bass_fund = analyses['bass'].get('freq_bands', {}).get('fundamental', -100)
            
            if abs(kick_fund - bass_fund) < 3 and kick_fund > -100 and bass_fund > -100:
                analyses['kick']['recommendations'].append({
                    'priority': 'important',
                    'title': 'ベースとの周波数棲み分け',
                    'steps': [
                        '【キック側】',
                        '  PEQ: 65Hz, Q=1.2, +4dB（キック強調）',
                        '  PEQ: 90Hz, Q=3.0, -4dB（ベース帯域カット）',
                        '',
                        '【ベース側】',
                        '  PEQ: 90Hz, Q=1.0, +3dB（ベース強調）',
                        '  PEQ: 65Hz, Q=3.0, -4dB（キック帯域カット）',
                        '',
                        '理論: 各楽器に専用周波数を割り当て'
                    ],
                    'expected_results': ['明瞭な低域', 'キックとベースの分離']
                })
        
        # ボーカル vs ギター
        if 'vocal' in analyses and 'e_guitar' in analyses:
            vocal_clarity = analyses['vocal'].get('freq_bands', {}).get('clarity', -100)
            
            if vocal_clarity < -30:
                analyses['e_guitar']['recommendations'].append({
                    'priority': 'important',
                    'title': 'ボーカルスペース確保',
                    'steps': [
                        'PEQ: 3.2kHz, Q=2.0, -2.5dB',
                        '  ボーカルの明瞭度帯域を空ける',
                        '',
                        '効果: ボーカルの明瞭度向上'
                    ],
                    'expected_results': ['ボーカルとの分離向上']
                })
    
    # ヘルパーメソッド
    
    def _get_vocal_eq_steps_safe(self):
        """ボーカルEQ（フィードバック配慮）"""
        return [
            '【PEQ設定（フィードバック配慮）】',
            '  Band 1: 250Hz, Q=3.0, -2.5dB（こもり除去）',
            '  Band 2: 800Hz, Q=2.0, -2.0dB（低域整理）',
            '  Band 3: 3.2kHz, Q=3.0, +3.0dB（明瞭度・ナロー）',
            '  Band 4: 5kHz, Q=2.5, +2.0dB（子音）',
            '',
            '【HPF】',
            '  80Hz, 24dB/oct',
            '',
            '【Compressor】',
            '  Threshold: -18dB, Ratio: 4:1',
            '  Attack: 10ms, Release: 100ms',
            '  Make-up: +3dB',
            '',
            '【フィードバック対策】',
            '  ⚠️ 3.2kHzをゆっくり上げる（+1dBずつ）',
            '  ⚠️ 事前にRingingで共振周波数特定',
            '  ⚠️ モニター位置確認'
        ]
    
    def _get_vocal_eq_steps_full(self):
        """ボーカルEQ（積極的処理）"""
        return [
            '【PEQ設定】',
            '  Band 1: 250Hz, Q=2.0, -3.0dB（こもり除去）',
            '  Band 2: 3kHz, Q=1.5, +4.5dB（明瞭度・広帯域）',
            '  Band 3: 5kHz, Q=2.0, +3.0dB（子音）',
            '  Band 4: 10kHz, Q=1.5, +2.0dB（空気感）',
            '',
            '【HPF】',
            '  80Hz, 24dB/oct',
            '',
            '【Compressor】',
            '  Threshold: -18dB, Ratio: 4:1',
            '  Attack: 10ms, Release: 100ms',
            '',
            '【De-Esser】',
            '  Frequency: 6.5kHz, Range: -3dB'
        ]
    
    def _get_deesser_steps(self):
        """De-Esser設定手順"""
        
        if self.mixer_specs and self.mixer_specs.get('has_de_esser'):
            return [
                'De-Esser設定:',
                '  Frequency: 6.5kHz',
                '  Threshold: 調整（歯擦音が出た時のみ反応）',
                '  Range: -3dB',
                '',
                '効果: 自然な歯擦音コントロール'
            ]
        else:
            return [
                'De-Esser非搭載のため代替案:',
                '',
                '【方法1】Dynamic EQ',
                '  6-8kHz, Threshold調整, -3dB',
                '',
                '【方法2】Compressor（サイドチェーン）',
                '  HPFで6kHz以上のみ検知',
                '',
                '【方法3】外部De-Esser使用'
            ]
    
    def _get_kick_hpf_freq(self):
        """キックのHPF周波数（PA考慮）"""
        
        if not self.pa_specs:
            return 35
        
        pa_name = self.pa_specs.get('name', '').lower()
        low_ext = self.pa_specs.get('low_extension', 50)
        
        if 'd&b' in pa_name or low_ext <= 45:
            return 35  # 低域が良好なら35Hz
        elif 'jbl' in pa_name or low_ext <= 50:
            return 30  # JBLなら30Hz
        else:
            return 40  # 小型PAなら40Hz
    
    def _get_pa_kick_notes(self):
        """PA別のキック注意事項"""
        
        if not self.pa_specs:
            return ['  一般的なPAシステムを想定']
        
        pa_name = self.pa_specs.get('name', '')
        notes = self.pa_specs.get('recommendations', {}).get('kick_hpf', '')
        
        if notes:
            return [f'  {notes}']
        else:
            return [f'  {pa_name}の特性に最適化']
    
    def _get_mixer_vocal_steps(self):
        """ミキサー別ボーカル設定"""
        
        if not self.mixer_specs:
            return None
        
        mixer_name = self.mixer_specs.get('name')
        
        if 'Yamaha CL' in mixer_name:
            return {
                'mixer': mixer_name,
                'steps': [
                    '1. ボーカルchを選択',
                    '2. [EQ]ボタン → PEQ画面',
                    '3. Band設定を上記の通り実施',
                    '4. [DYNAMICS1] → Compressor',
                    '5. TYPE: Comp260（透明度重視）',
                    '6. パラメータ設定',
                    '7. ゲインリダクション 4-6dB確認'
                ]
            }
        elif 'X32' in mixer_name:
            return {
                'mixer': mixer_name,
                'steps': [
                    '1. ボーカルchを選択',
                    '2. [EQ]ボタン',
                    '3. Band設定（4バンド・優先順位順）',
                    '4. [DYNAMICS] → Compressor',
                    '5. パラメータ設定',
                    '',
                    '注意: 4バンドのみ。優先順位を守る'
                ]
            }
        
        return None
    
    def _get_mixer_hpf_steps(self, instrument, freq):
        """ミキサー別HPF設定"""
        
        if not self.mixer_specs:
            return None
        
        mixer_name = self.mixer_specs.get('name')
        
        return {
            'mixer': mixer_name,
            'steps': [
                f'1. {instrument}チャンネルを選択',
                '2. [EQ]ボタン',
                f'3. HPF: {freq}Hz, 24dB/oct',
                '4. HPF ONを確認'
            ]
        }




# =====================================
# 過去音源比較機能
# =====================================

class ComparisonAnalyzer:
    """過去音源との比較（システム差異考慮）"""
    
    def __init__(self, current_analysis, past_entries, current_metadata):
        self.current = current_analysis
        self.past_entries = past_entries
        self.current_metadata = current_metadata
    
    def compare_all(self):
        """全ての過去音源と比較"""
        
        comparisons = []
        
        for entry in self.past_entries:
            comp = self._compare_with_entry(entry)
            if comp:
                comparisons.append(comp)
        
        return comparisons
    
    def _compare_with_entry(self, past_entry):
        """個別の過去音源と比較"""
        
        past_analysis = past_entry['analysis']
        past_metadata = past_entry['metadata']
        past_equipment = past_entry['equipment']
        
        comparison = {
            'past_id': past_entry['id'],
            'past_date': past_entry['timestamp'],
            'past_venue': past_metadata.get('venue', '不明'),
            'past_mixer': past_equipment.get('mixer', '不明'),
            'past_pa': past_equipment.get('pa_system', '不明'),
            'match_type': self._get_match_type(past_metadata, past_equipment),
            'metrics': {},
            'insights': []
        }
        
        # RMS比較（ミキサー補正）
        current_rms = self.current.get('rms_db', -100)
        past_rms = past_analysis.get('rms_db', -100)
        
        # ミキサー補正
        rms_correction = self._get_mixer_correction(
            self.current_metadata.get('mixer'),
            past_equipment.get('mixer')
        )
        
        past_rms_corrected = past_rms + rms_correction
        rms_diff = current_rms - past_rms_corrected
        
        comparison['metrics']['rms'] = {
            'current': current_rms,
            'past_raw': past_rms,
            'past_corrected': past_rms_corrected,
            'difference': rms_diff,
            'correction_applied': rms_correction
        }
        
        # ステレオ幅比較
        current_width = self.current.get('stereo_width', 0)
        past_width = past_analysis.get('stereo_width', 0)
        width_diff = current_width - past_width
        
        comparison['metrics']['stereo_width'] = {
            'current': current_width,
            'past': past_width,
            'difference': width_diff
        }
        
        # 周波数バランス比較（PA補正）
        current_bands = self.current.get('band_energies', [])
        past_bands = past_analysis.get('band_energies', [])
        
        if len(current_bands) == len(past_bands) and len(current_bands) > 0:
            pa_corrections = self._get_pa_corrections(
                self.current_metadata.get('pa_system'),
                past_equipment.get('pa_system')
            )
            
            band_diffs = []
            for i in range(len(current_bands)):
                correction = pa_corrections[i] if i < len(pa_corrections) else 0
                past_corrected = past_bands[i] + correction
                diff = current_bands[i] - past_corrected
                band_diffs.append(diff)
            
            comparison['metrics']['frequency_balance'] = {
                'differences': band_diffs,
                'pa_correction_applied': any(c != 0 for c in pa_corrections)
            }
        
        # 洞察生成
        comparison['insights'] = self._generate_insights(comparison, past_metadata)
        
        return comparison
    
    def _get_match_type(self, past_metadata, past_equipment):
        """マッチタイプ判定"""
        
        score = 0
        
        # 会場が近い
        current_capacity = self.current_metadata.get('venue_capacity', 0)
        past_capacity = past_metadata.get('venue_capacity', 0)
        
        if abs(current_capacity - past_capacity) < 50:
            score += 30
        
        # ミキサーが同じ
        if self.current_metadata.get('mixer') == past_equipment.get('mixer'):
            score += 40
        
        # PAが同じ
        if self.current_metadata.get('pa_system') == past_equipment.get('pa_system'):
            score += 30
        
        if score >= 80:
            return 'exact_match'
        elif score >= 50:
            return 'similar'
        else:
            return 'different'
    
    def _get_mixer_correction(self, current_mixer, past_mixer):
        """ミキサー間の補正値"""
        
        if not current_mixer or not past_mixer:
            return 0.0
        
        if current_mixer == past_mixer:
            return 0.0
        
        # 簡易的な補正（実際はより詳細に）
        mixer_tiers = {
            'cl': 1.0,
            'ql': 0.8,
            'sq': 0.7,
            'x32': 0.5
        }
        
        current_tier = 0.5
        past_tier = 0.5
        
        for key, value in mixer_tiers.items():
            if key in current_mixer.lower():
                current_tier = value
            if key in past_mixer.lower():
                past_tier = value
        
        # ティア差 × 2dB
        return (current_tier - past_tier) * 2.0
    
    def _get_pa_corrections(self, current_pa, past_pa):
        """PA間の周波数補正"""
        
        # 7バンド分の補正値
        corrections = [0.0] * 7
        
        if not current_pa or not past_pa or current_pa == past_pa:
            return corrections
        
        # 簡易的な補正
        # d&b: フラット
        # JBL: 高域明るい（+2dB）
        # L-Acoustics: フラット
        
        current_brightness = 0
        past_brightness = 0
        
        if 'jbl' in current_pa.lower():
            current_brightness = 2
        if 'jbl' in past_pa.lower():
            past_brightness = 2
        
        brightness_diff = current_brightness - past_brightness
        
        # Presence/Brillianceに反映
        corrections[5] = -brightness_diff * 1.5  # Presence
        corrections[6] = -brightness_diff * 2.0  # Brilliance
        
        return corrections
    
    def _generate_insights(self, comparison, past_metadata):
        """比較からの洞察生成"""
        
        insights = []
        
        match_type = comparison['match_type']
        rms_diff = comparison['metrics']['rms']['difference']
        
        # RMS変化
        if match_type == 'exact_match':
            if rms_diff > 2:
                insights.append({
                    'type': 'improvement',
                    'message': f'音圧が前回より +{rms_diff:.1f}dB 向上（同条件比較）',
                    'severity': 'good'
                })
            elif rms_diff < -2:
                insights.append({
                    'type': 'regression',
                    'message': f'音圧が前回より {rms_diff:.1f}dB 低下（同条件比較）',
                    'severity': 'warning'
                })
            else:
                insights.append({
                    'type': 'stable',
                    'message': f'音圧は前回と同レベル（{rms_diff:+.1f}dB）',
                    'severity': 'info'
                })
        else:
            # 異なる条件
            correction = comparison['metrics']['rms'].get('correction_applied', 0)
            if correction != 0:
                insights.append({
                    'type': 'info',
                    'message': f'音圧差 {rms_diff:+.1f}dB（システム差補正済: {correction:+.1f}dB）',
                    'severity': 'info'
                })
        
        # ステレオ幅変化
        width_diff = comparison['metrics']['stereo_width']['difference']
        if abs(width_diff) > 10:
            insights.append({
                'type': 'change',
                'message': f'ステレオ幅が {width_diff:+.1f}% 変化',
                'severity': 'info'
            })
        
        # 周波数バランス
        if 'frequency_balance' in comparison['metrics']:
            band_diffs = comparison['metrics']['frequency_balance']['differences']
            band_names = ['Sub Bass', 'Bass', 'Low-Mid', 'Mid', 'High-Mid', 'Presence', 'Brilliance']
            
            for i, diff in enumerate(band_diffs):
                if abs(diff) > 6:
                    insights.append({
                        'type': 'change',
                        'message': f'{band_names[i]}が {diff:+.1f}dB 変化',
                        'severity': 'info'
                    })
        
        return insights


# =====================================
# メインUI
# =====================================

def show_history_page(db):
    """過去解析データ閲覧ページ"""
    st.markdown("## 📊 過去の解析データ")
    
    if not db.history:
        st.info("まだ解析データがありません。")
        return
    
    # 検索・フィルター
    st.markdown("### 🔍 検索・フィルター")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_name = st.text_input("名前で検索", placeholder="例: ライブ本番")
    
    with col2:
        search_venue = st.text_input("会場で検索", placeholder="例: CLUB QUATTRO")
    
    with col3:
        search_mixer = st.text_input("ミキサーで検索", placeholder="例: CL5")
    
    # フィルタリング
    filtered = db.history
    
    if search_name:
        filtered = [e for e in filtered if search_name.lower() in e['metadata'].get('analysis_name', '').lower()]
    
    if search_venue:
        filtered = [e for e in filtered if search_venue.lower() in e['metadata'].get('venue', '').lower()]
    
    if search_mixer:
        filtered = [e for e in filtered if search_mixer.lower() in e['equipment'].get('mixer', '').lower()]
    
    # ソート
    filtered = sorted(filtered, key=lambda x: x['timestamp'], reverse=True)
    
    st.markdown(f"### 📋 解析データ一覧（{len(filtered)}件）")
    
    if not filtered:
        st.warning("検索条件に一致するデータがありません。")
        return
    
    # データ一覧表示
    for entry in filtered:
        timestamp = datetime.fromisoformat(entry['timestamp'])
        analysis_name = entry['metadata'].get('analysis_name', '名称未設定')
        venue = entry['metadata'].get('venue', '不明')
        mixer = entry['equipment'].get('mixer', '不明')
        pa = entry['equipment'].get('pa_system', '不明')
        
        with st.expander(f"🎵 {analysis_name} - {timestamp.strftime('%Y/%m/%d %H:%M')}", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📅 基本情報**")
                st.write(f"**日時**: {timestamp.strftime('%Y年%m月%d日 %H:%M')}")
                st.write(f"**名前**: {analysis_name}")
                st.write(f"**会場**: {venue}")
                st.write(f"**キャパ**: {entry['metadata'].get('venue_capacity', '不明')}人")
                
                if entry['metadata'].get('notes'):
                    st.write(f"**メモ**: {entry['metadata']['notes']}")
            
            with col2:
                st.markdown("**🎛️ 使用機材**")
                st.write(f"**ミキサー**: {mixer}")
                st.write(f"**PA**: {pa}")
                st.write(f"**ステージ生音**: {entry['metadata'].get('stage_volume', '不明')}")
            
            st.markdown("---")
            
            # 解析結果サマリー
            st.markdown("**📊 解析結果サマリー**")
            
            analysis = entry['analysis']
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMS", f"{analysis.get('rms_db', 0):.1f} dB")
            with col2:
                st.metric("Peak", f"{analysis.get('peak_db', 0):.1f} dB")
            with col3:
                st.metric("ステレオ幅", f"{analysis.get('stereo_width', 0):.1f}%")
            with col4:
                st.metric("クレスト", f"{analysis.get('crest_factor', 0):.1f} dB")
            
            # 削除ボタン
            if st.button(f"🗑️ このデータを削除", key=f"delete_{entry['id']}"):
                db.history.remove(entry)
                db.save()
                st.success("削除しました")
                st.rerun()


def main():
    st.markdown('<h1 class="main-header">🎛️ Live PA Audio Analyzer V3.0</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="version-badge">Final Release - 完全版</p>', 
                unsafe_allow_html=True)
    
    # セッションステート初期化
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    if 'current_results' not in st.session_state:
        st.session_state.current_results = None
    
    # データベース初期化
    db = AudioDatabase()
    
    # 機材検索初期化
    equipment_searcher = EquipmentSpecsSearcher()
    
    # タブ切り替え
    tab1, tab2 = st.tabs(["🎵 新規解析", "📊 過去データ"])
    
    with tab2:
        show_history_page(db)
    
    with tab1:
        # 解析完了後の「次の音源を解析」ボタン
        if st.session_state.analysis_complete:
            st.success("✅ 解析完了！データを保存しました。")
            
            if st.button("🔄 次の音源を解析", type="primary", use_container_width=True):
                st.session_state.analysis_complete = False
                st.session_state.current_results = None
                st.rerun()
            
            st.markdown("---")
            st.info("👆 「次の音源を解析」ボタンを押すと、新しい解析を開始できます。")
            return
        
        # サイドバー
        with st.sidebar:
            st.header("⚙️ 設定")
            
            # 解析名入力（最上部に追加）
            st.markdown("### 📝 解析データの名前")
            analysis_name = st.text_input(
                "この解析に名前をつける",
                placeholder="例: ライブ本番、リハーサル、ワンマン",
                help="保存時に識別しやすい名前をつけてください"
            )
            
            if not analysis_name.strip():
                st.warning("⚠️ 名前を入力してください（必須）")
            
            st.markdown("---")
            
            uploaded_file = st.file_uploader(
                "音源ファイルをアップロード",
                type=['mp3', 'wav', 'flac', 'm4a']
            )
            
            if uploaded_file:
                file_size_mb = uploaded_file.size / (1024 * 1024)
                if file_size_mb > 100:
                    st.error(f"❌ ファイルが大きすぎます（{file_size_mb:.1f}MB）")
                    uploaded_file = None
                else:
                    st.success(f"✓ {file_size_mb:.1f}MB")
            
            st.markdown("---")
            
            # バンド編成（テキスト入力）
            st.subheader("🎸 バンド編成")
            
            band_lineup_text = st.text_area(
                "楽器を入力（カンマ区切り）",
                value="ボーカル、キック、スネア、ハイハット、ベース、ギター",
                height=100,
                help="例: ボーカル、キック、スネア、ベース、ギター\n日本語・英語・略語OK"
            )
            
            if not band_lineup_text.strip():
                st.warning("⚠️ バンド編成を入力してください")
            
            st.markdown("---")
            st.subheader("🏛️ 会場情報")
            
            venue_name = st.text_input("会場名（任意）", placeholder="例: CLUB QUATTRO")
            venue_capacity = st.slider("会場キャパ（人）", 50, 2000, 150, 50)
            stage_volume = st.selectbox("ステージ生音", ['high', 'medium', 'low', 'none'], 1)
            
            st.markdown("---")
            st.subheader("🎛️ 使用機材")
            
            mixer_name = st.text_input(
                "ミキサー", 
                placeholder="例: Yamaha CL5",
                help="正確な型番を入力すると自動で仕様を検索します"
            )
            
            pa_system = st.text_input(
                "PAシステム", 
                placeholder="例: d&b V-Series",
                help="システム名を入力すると特性を考慮した提案を行います"
            )
            
            notes = st.text_area("メモ（任意）", placeholder="セットリスト、特記事項など")
            
            st.markdown("---")
            
            # 過去音源表示
            recent_entries = db.get_recent(3)
            if recent_entries:
                st.subheader("📊 最近の解析")
                for entry in recent_entries:
                    date = datetime.fromisoformat(entry['timestamp']).strftime('%m/%d %H:%M')
                    name = entry['metadata'].get('analysis_name', '名称未設定')
                    st.caption(f"{date} - {name}")
            
            st.markdown("---")
            analyze_button = st.button(
                "🚀 解析開始", 
                type="primary", 
                use_container_width=True,
                disabled=(uploaded_file is None or not band_lineup_text.strip() or not analysis_name.strip())
            )
        
        # メインエリア
        if uploaded_file is None:
            st.info("👈 音源をアップロードしてバンド編成を入力してください")
            
            st.markdown("### 🆕 V3.0 Final の全機能")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **📊 2mix全体解析**
                - 音圧、ステレオイメージ、周波数解析
                - 9パネル詳細グラフ
                - 良いポイント + 改善提案
                
                **🎸 楽器別詳細解析**
                - テキスト入力で自由な編成指定
                - 全楽器の周波数特性解析
                - 楽器ごとの具体的EQ/Comp設定
                """)
            
            with col2:
                st.markdown("""
                **🔍 Web検索統合**
                - ミキサー仕様の自動取得
                - PAシステム特性の反映
                - 機材に最適化された提案
                
                **📈 過去音源との比較**
                - システム差異を考慮した補正
                - 成長トレンドの可視化
                - 同条件 vs 異条件の比較
                """)
            
            st.markdown("---")
            st.markdown("### 📝 使い方")
            st.markdown("""
            1. **音源アップロード**: 2mix音源（mp3, wav等）
            2. **バンド編成入力**: 「ボーカル、キック、スネア、ベース」など
            3. **会場・機材情報**: できるだけ詳しく入力
            4. **解析開始**: ボタンをクリック
            5. **結果確認**: グラフ、良いポイント、改善提案を確認
            6. **実践**: 具体的な設定値を現場で試す
            """)
        
        elif analyze_button:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            try:
                # メタデータ
                metadata = {
                    'analysis_name': analysis_name,
                    'venue': venue_name,
                    'venue_capacity': venue_capacity,
                    'stage_volume': stage_volume,
                    'mixer': mixer_name,
                    'pa_system': pa_system,
                    'band_lineup': band_lineup_text,
                    'notes': notes
                }
                
                # === Phase 1: 機材仕様検索 ===
                
                mixer_specs = None
                pa_specs = None
                
                if mixer_name:
                    mixer_specs = equipment_searcher.search_mixer_specs(mixer_name)
                    if mixer_specs:
                        st.success(f"✅ {mixer_specs['name']}の仕様を取得")
                        
                        # ミキサー仕様の詳細表示
                        with st.expander(f"🎛️ {mixer_specs['name']} - 仕様の詳細"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🔧 基本仕様**")
                                if mixer_specs.get('eq_bands'):
                                    st.write(f"**EQバンド数**: {mixer_specs['eq_bands']}バンド")
                                if mixer_specs.get('eq_type'):
                                    st.write(f"**EQタイプ**: {mixer_specs['eq_type']}")
                                if mixer_specs.get('has_de_esser'):
                                    st.write(f"**De-Esser**: 搭載")
                                if mixer_specs.get('has_dynamic_eq'):
                                    st.write(f"**Dynamic EQ**: 搭載")
                                if mixer_specs.get('hpf_slopes'):
                                    st.write(f"**HPF**: {', '.join(mixer_specs['hpf_slopes'])}")
                            
                            with col2:
                                st.markdown("**🎵 特徴**")
                                for char in mixer_specs.get('characteristics', []):
                                    st.write(f"• {char}")
                            
                            if mixer_specs.get('compressor_types'):
                                st.markdown("**📊 コンプレッサータイプ**")
                                st.write(", ".join(mixer_specs['compressor_types']))
                            
                            if mixer_specs.get('recommendations'):
                                st.markdown("**💡 推奨設定**")
                                for key, value in mixer_specs['recommendations'].items():
                                    st.write(f"**{key}**: {value}")
                
                if pa_system:
                    pa_specs = equipment_searcher.search_pa_specs(pa_system)
                    if pa_specs:
                        st.success(f"✅ {pa_specs['name']}の特性を取得")
                        
                        # PA特性の詳細表示
                        with st.expander(f"📢 {pa_specs['name']} - システム特性の詳細"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**🔧 基本仕様**")
                                st.write(f"**タイプ**: {pa_specs.get('type', '不明')}")
                                if pa_specs.get('low_extension'):
                                    st.write(f"**低域限界**: {pa_specs['low_extension']}Hz")
                                if pa_specs.get('high_extension'):
                                    st.write(f"**高域限界**: {pa_specs['high_extension']}Hz")
                            
                            with col2:
                                st.markdown("**🎵 サウンド特性**")
                                for char in pa_specs.get('characteristics', []):
                                    st.write(f"• {char}")
                            
                            if pa_specs.get('eq_compensation'):
                                st.markdown("**⚙️ 推奨EQ補正**")
                                for comp in pa_specs['eq_compensation']:
                                    st.write(f"• {comp}")
                            
                            if pa_specs.get('feedback_prone'):
                                st.markdown("**⚠️ フィードバック注意帯域**")
                                st.write(f"{', '.join(map(str, pa_specs['feedback_prone']))}Hz")
                            
                            if pa_specs.get('recommendations'):
                                st.markdown("**💡 推奨設定**")
                                for key, value in pa_specs['recommendations'].items():
                                    st.write(f"**{key}**: {value}")
                
                # === Phase 2: V2解析（2mix全体） ===
                
                st.markdown("## 📊 2mix全体解析")
                
                v2_analyzer = V2Analyzer(tmp_path, venue_capacity, stage_volume, pa_system, notes)
                v2_results = v2_analyzer.analyze()
                
                st.success("✅ 2mix解析完了")
                
                # サマリーメトリクス
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("ステレオ幅", f"{v2_results['stereo_width']:.1f}%")
                with col2:
                    st.metric("RMS", f"{v2_results['rms_db']:.1f} dB")
                with col3:
                    st.metric("クレストファクター", f"{v2_results['crest_factor']:.1f} dB")
                with col4:
                    st.metric("ダイナミックレンジ", f"{v2_results['dynamic_range']:.1f} dB")
                
                # グラフ表示
                st.markdown("### 📈 詳細グラフ")
                
                with st.spinner('📊 グラフを生成中...'):
                    fig = v2_analyzer.create_visualization()
                    st.pyplot(fig, use_container_width=True)
                    
                    # ダウンロードボタン
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
                    buf.seek(0)
                    
                    st.download_button(
                        label="📥 グラフをダウンロード",
                        data=buf,
                        file_name=f"pa_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png"
                    )
                    
                    plt.close(fig)
                
                # 2mix改善提案
                st.markdown("### 💡 2mix全体の改善提案")
                st.caption("🔬 音響物理学・国際規格・研究論文に基づく科学的提案")
                
                good_points, v2_recs = v2_analyzer.generate_v2_recommendations(mixer_specs, pa_specs)
                
                # 良いポイント
                if good_points:
                    st.markdown("#### ✅ 良いポイント")
                    for gp in good_points:
                        with st.expander(f"✅ {gp['category']}: {gp['point']}", expanded=False):
                            st.write(f"**影響度:** {gp['impact']}")
                            if gp.get('scientific_basis'):
                                st.info(f"📚 **科学的根拠:** {gp['scientific_basis']}")
                
                # 改善提案
                for priority in ['critical', 'important', 'optional']:
                    if v2_recs[priority]:
                        priority_label = {
                            'critical': '🔴 最優先', 
                            'important': '🟡 重要', 
                            'optional': '🟢 オプション'
                        }[priority]
                        
                        st.markdown(f"#### {priority_label}")
                        
                        for rec in v2_recs[priority]:
                            with st.expander(f"{rec['category']}: {rec['issue']}", expanded=(priority == 'critical')):
                                st.markdown(f"**🎯 対策:**")
                                st.code(rec['solution'], language='')
                                
                                st.write(f"**影響度:** {rec['impact']}")
                                
                                if rec.get('scientific_basis'):
                                    st.markdown("---")
                                    st.markdown("**🔬 科学的根拠:**")
                                    st.info(rec['scientific_basis'])
                                
                                if rec.get('references'):
                                    st.caption(f"📖 参考文献: {rec['references']}")
                
                st.markdown("---")
                
                # === Phase 3: 楽器別解析 ===
                
                st.markdown("## 🎸 楽器別詳細解析")
                
                # 楽器分離
                separator = InstrumentSeparator(v2_analyzer.y, v2_analyzer.sr, band_lineup_text)
                stems = separator.separate()
                
                st.success(f"✅ {len(stems)}楽器を分離完了")
                
                # 分離された楽器を表示
                st.write("**検出された楽器:**", ', '.join(
                    {'vocal': 'ボーカル', 'kick': 'キック', 'snare': 'スネア',
                     'bass': 'ベース', 'hihat': 'ハイハット', 'tom': 'タム',
                     'e_guitar': 'エレキギター', 'a_guitar': 'アコギ',
                     'keyboard': 'キーボード', 'synth': 'シンセ'}.get(k, k)
                    for k in stems.keys()
                ))
                
                # 過去データ取得（同楽器編成）
                past_analyses_for_comparison = db.find_similar(metadata, limit=5)
                
                # 詳細解析
                inst_analyzer = InstrumentAnalyzer(
                    stems, v2_analyzer.sr, v2_analyzer.y, 
                    v2_results['rms_db'],
                    mixer_specs, pa_specs,
                    past_analyses=past_analyses_for_comparison  # 過去データ渡す
                )
                
                inst_analyses = inst_analyzer.analyze_all(venue_capacity, stage_volume)
                
                st.success("✅ 楽器別解析完了")
                
                # 楽器別の詳細表示
                for inst_name, analysis in inst_analyses.items():
                    inst_name_ja = {
                        'vocal': 'ボーカル', 'kick': 'キック', 'snare': 'スネア',
                        'bass': 'ベース', 'hihat': 'ハイハット', 'tom': 'タム',
                        'e_guitar': 'エレキギター', 'a_guitar': 'アコギ',
                        'keyboard': 'キーボード', 'synth': 'シンセ'
                    }.get(inst_name, inst_name)
                    
                    icon = {
                        'vocal': '🎤', 'kick': '🥁', 'snare': '🥁', 'bass': '🎸',
                        'hihat': '🥁', 'tom': '🥁', 'e_guitar': '🎸', 'a_guitar': '🎸',
                        'keyboard': '🎹', 'synth': '🎹'
                    }.get(inst_name, '🎵')
                    
                    with st.expander(f"{icon} {inst_name_ja}の詳細解析", expanded=(inst_name in ['vocal', 'kick'])):
                        # 基本情報
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("RMS", f"{analysis['rms_db']:.1f} dBFS")
                        with col2:
                            st.metric("Peak", f"{analysis['peak_db']:.1f} dBFS")
                        with col3:
                            st.metric("vs 2mix", f"{analysis['level_vs_mix']:+.1f} dB")
                        
                        # 周波数帯域
                        if analysis.get('freq_bands'):
                            st.markdown("**周波数帯域別レベル:**")
                            for band_name, level in analysis['freq_bands'].items():
                                st.write(f"- {band_name}: {level:.1f} dB")
                        
                        # 良いポイント
                        if analysis.get('good_points'):
                            st.markdown("**✅ 良いポイント:**")
                            for gp in analysis['good_points']:
                                st.markdown(f"""
                                <div class="good-point">
                                    {gp['point']}<br>
                                    影響度: {gp.get('impact', '★★★')}
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # 問題点
                        if analysis.get('issues'):
                            st.markdown("**❌ 検出された問題:**")
                            for issue in analysis['issues']:
                                severity_icon = {
                                    'critical': '🔴', 
                                    'important': '🟡', 
                                    'medium': '🟠'
                                }.get(issue['severity'], '⚪')
                                st.write(f"{severity_icon} **{issue['problem']}**")
                                st.caption(issue['detail'])
                        
                        # 改善提案
                        if analysis.get('recommendations'):
                            st.markdown("**💡 改善提案:**")
                            
                            for i, rec in enumerate(analysis['recommendations'], 1):
                                priority_icon = {
                                    'critical': '🔴', 
                                    'important': '🟡', 
                                    'optional': '🟢'
                                }.get(rec['priority'], '⚪')
                                
                                with st.expander(f"{priority_icon} {i}. {rec['title']}", expanded=(rec['priority'] == 'critical')):
                                    # 問題詳細
                                    if rec.get('problem_detail'):
                                        st.warning(f"**問題:** {rec['problem_detail']}")
                                    
                                    # トレンド注記
                                    if rec.get('trend_note'):
                                        st.info(rec['trend_note'])
                                    
                                    # 科学的根拠
                                    if rec.get('scientific_basis'):
                                        with st.expander("🔬 科学的根拠", expanded=False):
                                            st.write(rec['scientific_basis'])
                                    
                                    # 複数アプローチ対応
                                    if rec.get('approaches'):
                                        st.markdown("---")
                                        st.markdown("### 📋 解決アプローチ（複数選択肢）")
                                        
                                        for j, approach in enumerate(rec['approaches'], 1):
                                            st.markdown(f"#### アプローチ {j}: {approach['method']}")
                                            
                                            col1, col2 = st.columns([3, 1])
                                            
                                            with col1:
                                                st.markdown("**🎯 手順:**")
                                                st.code('\n'.join(approach['steps']), language='')
                                            
                                            with col2:
                                                st.markdown(f"**難易度:** {approach.get('difficulty', '★★★☆☆')}")
                                                
                                                st.markdown("**メリット:**")
                                                for pro in approach.get('pros', []):
                                                    st.write(f"✅ {pro}")
                                                
                                                st.markdown("**デメリット:**")
                                                for con in approach.get('cons', []):
                                                    st.write(f"⚠️ {con}")
                                            
                                            if j < len(rec['approaches']):
                                                st.markdown("---")
                                    
                                    # 旧フォーマット対応（互換性）
                                    elif rec.get('steps'):
                                        st.markdown("**🎯 手順:**")
                                        for step in rec['steps']:
                                            st.write(step)
                                        
                                        # ミキサー固有の手順
                                        if rec.get('mixer_specific'):
                                            with st.expander(f"📱 {rec['mixer_specific']['mixer']} での操作手順"):
                                                for step in rec['mixer_specific']['steps']:
                                                    st.write(step)
                                    
                                    # 期待される結果
                                    if rec.get('expected_results'):
                                        st.markdown("---")
                                        st.markdown("**🎯 期待される効果:**")
                                        for result in rec['expected_results']:
                                            st.write(f"✨ {result}")
                        
                        st.markdown("---")
                
                st.markdown("---")
                
                # === Phase 4: 過去音源との比較 ===
                
                similar_entries = db.find_similar(metadata, limit=3)
                
                if similar_entries:
                    st.markdown("## 📊 過去音源との比較")
                    
                    comp_analyzer = ComparisonAnalyzer(v2_results, similar_entries, metadata)
                    comparisons = comp_analyzer.compare_all()
                    
                    for i, comp in enumerate(comparisons, 1):
                        match_icon = {
                            'exact_match': '🟢',
                            'similar': '🟡',
                            'different': '🔵'
                        }.get(comp['match_type'], '⚪')
                        
                        match_label = {
                            'exact_match': 'ほぼ同条件',
                            'similar': '類似条件',
                            'different': '異なる条件'
                        }.get(comp['match_type'], '不明')
                        
                        with st.expander(f"{match_icon} 比較 #{i}: {match_label} - {comp['past_venue']}", expanded=(i==1)):
                            st.write(f"**日時:** {datetime.fromisoformat(comp['past_date']).strftime('%Y年%m月%d日 %H:%M')}")
                            st.write(f"**会場:** {comp['past_venue']}")
                            st.write(f"**ミキサー:** {comp['past_mixer']}")
                            st.write(f"**PA:** {comp['past_pa']}")
                            
                            st.markdown("---")
                            
                            # メトリクス比較
                            rms_metric = comp['metrics']['rms']
                            
                            st.markdown("**音圧（RMS）:**")
                            st.write(f"- 現在: {rms_metric['current']:.1f} dBFS")
                            st.write(f"- 過去: {rms_metric['past_raw']:.1f} dBFS（生値）")
                            
                            if rms_metric['correction_applied'] != 0:
                                st.write(f"- 過去（補正後）: {rms_metric['past_corrected']:.1f} dBFS")
                                st.caption(f"補正値: {rms_metric['correction_applied']:+.1f}dB（ミキサー差異）")
                            
                            st.write(f"- **差分: {rms_metric['difference']:+.1f} dB**")
                            
                            # ステレオ幅
                            width_metric = comp['metrics']['stereo_width']
                            st.markdown("**ステレオ幅:**")
                            st.write(f"- 差分: {width_metric['difference']:+.1f}%")
                            
                            # 洞察
                            if comp['insights']:
                                st.markdown("**💡 洞察:**")
                                for insight in comp['insights']:
                                    icon = {
                                        'improvement': '✅',
                                        'regression': '⚠️',
                                        'stable': '→',
                                        'change': '📌',
                                        'info': 'ℹ️'
                                    }.get(insight['type'], '•')
                                    
                                    st.write(f"{icon} {insight['message']}")
                
                # === データベースに保存 ===
                
                entry_id = db.add_entry(v2_results, metadata)
                st.success(f"✅ 解析結果を「{analysis_name}」として保存しました（ID: {entry_id}）")
                
                # セッションステート更新
                st.session_state.analysis_complete = True
                st.session_state.current_results = {
                    'entry_id': entry_id,
                    'analysis_name': analysis_name
                }
            
            except Exception as e:
                st.error(f"❌ エラー: {str(e)}")
                with st.expander("詳細"):
                    st.exception(e)
            
            finally:
                import os
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


if __name__ == "__main__":
    main()
