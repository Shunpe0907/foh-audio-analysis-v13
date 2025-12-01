"""
PA Audio Analyzer V3.0 - 認証システム統合版

機能:
- ユーザー認証（ログイン・新規登録）
- ユーザー別データ管理
- 過去解析データの閲覧・比較
- 管理者ダッシュボード

使い方:
    streamlit run pa_analyzer_with_auth.py
"""

import streamlit as st
import sys
from pathlib import Path

# 認証システムをインポート
from auth_system import (
    UserDatabase, UserAudioDatabase,
    init_session_state,
    show_login_page, show_register_page,
    show_user_profile, show_admin_dashboard
)

# ページ設定
st.set_page_config(
    page_title="PA Audio Analyzer V3.0",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS（既存のものを使用）
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 0.5rem;
}
.version-badge {
    text-align: center;
    color: #666;
    font-size: 0.9rem;
    margin-bottom: 2rem;
}
.good-point {
    background-color: #d4edda;
    border-left: 4px solid #28a745;
    padding: 1rem;
    margin: 0.5rem 0;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


def main():
    """メインアプリケーション"""
    
    # セッションステート初期化
    init_session_state()
    
    # データベース初期化
    user_db = UserDatabase()
    audio_db = UserAudioDatabase()
    
    # 認証チェック
    if not st.session_state.authenticated:
        # ログイン・登録ページ
        if st.session_state.page == 'login':
            show_login_page(user_db)
        elif st.session_state.page == 'register':
            show_register_page(user_db)
        
        # 説明
        st.markdown("---")
        st.markdown("""
        ## 🎛️ PA Audio Analyzer V3.0 について
        
        ライブPA用の2mixおよび楽器別オーディオ解析ツールです。
        
        ### 主な機能
        
        - **2mix全体解析**: 音圧、ステレオイメージ、周波数バランス
        - **楽器別詳細解析**: ボーカル、ドラム、ベース、ギターなど
        - **科学的根拠に基づく提案**: ITU-R、ISO、AES規格準拠
        - **過去データ比較**: 成長トレンドの可視化
        - **機材別最適化**: ミキサー・PAシステム特性を考慮
        
        ### ログイン・登録について
        
        - **新規ユーザー**: 「新規登録」から無料でアカウント作成
        - **解析履歴**: ログインすることで過去の解析データを保存・比較可能
        - **プライバシー**: データは個別管理、他のユーザーからは見えません
        """)
        
        return
    
    # ログイン済み
    user = st.session_state.user
    
    # サイドバー
    with st.sidebar:
        # ユーザー情報表示
        if user['role'] == 'admin':
            st.markdown(f"### 🛡️ 管理者: {user['name']}")
        else:
            st.markdown(f"### 👤 {user['name']}")
        
        st.caption(f"📧 {user['email']}")
        
        st.markdown("---")
        
        # メニュー
        menu = st.radio(
            "メニュー",
            ["🎵 音源解析", "📊 過去データ", "👤 プロフィール"] +
            (["🛡️ 管理者ダッシュボード"] if user['role'] == 'admin' else []),
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # ログアウト
        if st.button("🚪 ログアウト", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.page = 'login'
            st.rerun()
    
    # メインコンテンツ
    if menu == "🎵 音源解析":
        show_analyzer_page(user, user_db, audio_db)
    
    elif menu == "📊 過去データ":
        show_history_page(user, audio_db)
    
    elif menu == "👤 プロフィール":
        show_user_profile(user_db)
    
    elif menu == "🛡️ 管理者ダッシュボード" and user['role'] == 'admin':
        show_admin_dashboard(user_db, audio_db)


def show_analyzer_page(user, user_db, audio_db):
    """音源解析ページ（メインPA Analyzerを統合）"""
    
    st.markdown('<h1 class="main-header">🎛️ Live PA Audio Analyzer V3.0</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="version-badge">Final Release - 認証システム統合版</p>', 
                unsafe_allow_html=True)
    
    st.info("""
    💡 **統合版の注意点**
    
    このファイルは認証システムのデモ版です。
    実際の解析機能を使用するには、`pa_analyzer_v3_final.py`の内容を
    この下に統合する必要があります。
    
    統合方法は `AUTH_INTEGRATION_GUIDE.md` を参照してください。
    """)
    
    # TODO: ここに pa_analyzer_v3_final.py の解析ロジックを統合
    # 現在はプレースホルダー
    
    st.markdown("---")
    st.markdown("### 📝 解析実行（プレースホルダー）")
    
    with st.form("demo_analysis"):
        analysis_name = st.text_input("解析名", placeholder="例: ライブ本番")
        venue_name = st.text_input("会場名", placeholder="例: CLUB QUATTRO")
        
        if st.form_submit_button("デモ解析実行"):
            # デモデータ保存
            demo_data = {
                'rms_db': -18.2,
                'peak_db': -3.1,
                'stereo_width': 65.3,
                'crest_factor': 12.3,
                'band_energies': []
            }
            
            demo_metadata = {
                'analysis_name': analysis_name or "デモ解析",
                'venue': venue_name or "デモ会場",
                'venue_capacity': 150,
                'mixer': 'Yamaha CL5',
                'pa_system': 'd&b V-Series',
                'band_lineup': 'ボーカル、キック、スネア、ベース、ギター',
                'notes': 'デモデータ'
            }
            
            # ユーザー統計更新
            user_db.update_user_stats(user['email'])
            
            # データ保存
            entry_id = audio_db.add_analysis(user['email'], demo_data, demo_metadata)
            
            st.success(f"✅ デモ解析完了！（ID: {entry_id}）")
            st.info("実際の解析機能は pa_analyzer_v3_final.py を統合してください")


def show_history_page(user, audio_db):
    """過去解析データページ"""
    
    st.markdown("## 📊 過去の解析データ")
    
    # ユーザーの解析データ取得
    analyses = audio_db.get_user_analyses(user['email'])
    
    if not analyses:
        st.info("まだ解析データがありません。「音源解析」から解析を実行してください。")
        return
    
    st.write(f"**総解析数: {len(analyses)}件**")
    
    # 検索・フィルター
    search = st.text_input("🔍 検索", placeholder="解析名、会場名で検索")
    
    # フィルタリング
    if search:
        filtered = [
            a for a in analyses
            if search.lower() in a['metadata'].get('analysis_name', '').lower()
            or search.lower() in a['metadata'].get('venue', '').lower()
        ]
    else:
        filtered = analyses
    
    st.write(f"**表示: {len(filtered)}件**")
    
    # データ一覧
    for analysis in filtered:
        from datetime import datetime
        
        timestamp = datetime.fromisoformat(analysis['timestamp'])
        name = analysis['metadata'].get('analysis_name', '名称未設定')
        venue = analysis['metadata'].get('venue', '不明')
        
        with st.expander(f"🎵 {name} - {venue} ({timestamp.strftime('%Y/%m/%d %H:%M')})", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📅 基本情報**")
                st.write(f"**解析日時**: {timestamp.strftime('%Y年%m月%d日 %H:%M')}")
                st.write(f"**解析名**: {name}")
                st.write(f"**会場**: {venue}")
                st.write(f"**キャパ**: {analysis['metadata'].get('venue_capacity', '不明')}人")
            
            with col2:
                st.markdown("**🎛️ 機材情報**")
                st.write(f"**ミキサー**: {analysis['metadata'].get('mixer', '不明')}")
                st.write(f"**PA**: {analysis['metadata'].get('pa_system', '不明')}")
                st.write(f"**バンド編成**: {analysis['metadata'].get('band_lineup', '不明')}")
            
            # 解析結果
            st.markdown("---")
            st.markdown("**📊 解析結果**")
            
            analysis_data = analysis.get('analysis', {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("RMS", f"{analysis_data.get('rms_db', 0):.1f} dB")
            with col2:
                st.metric("Peak", f"{analysis_data.get('peak_db', 0):.1f} dB")
            with col3:
                st.metric("ステレオ幅", f"{analysis_data.get('stereo_width', 0):.1f}%")
            with col4:
                st.metric("クレスト", f"{analysis_data.get('crest_factor', 0):.1f} dB")
            
            # メモ
            if analysis['metadata'].get('notes'):
                st.markdown("**📝 メモ**")
                st.write(analysis['metadata']['notes'])
            
            # 削除ボタン
            if st.button(f"🗑️ このデータを削除", key=f"delete_{analysis['id']}"):
                if audio_db.delete_analysis(user['email'], analysis['id']):
                    st.success("削除しました")
                    st.rerun()
                else:
                    st.error("削除に失敗しました")


if __name__ == "__main__":
    main()
