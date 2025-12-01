"""
PA Audio Analyzer - ユーザー認証・管理システム

機能:
1. ユーザー登録・ログイン
2. ユーザー別データ管理
3. 管理者ダッシュボード
4. セキュアなパスワード管理
"""

import streamlit as st
import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
import secrets


class UserDatabase:
    """ユーザーデータベース管理"""
    
    def __init__(self, db_path='users.json'):
        self.db_path = Path(db_path)
        self.users = {}
        self.load()
    
    def load(self):
        """ユーザーデータ読み込み"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.users = json.load(f)
            except:
                self.users = {}
        else:
            # 初回起動時: デフォルト管理者アカウント作成
            self.create_default_admin()
    
    def save(self):
        """ユーザーデータ保存"""
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, ensure_ascii=False, indent=2)
    
    def create_default_admin(self):
        """デフォルト管理者アカウント作成"""
        admin_email = "admin@pa-analyzer.local"
        admin_password = "admin123"  # 初回ログイン後に変更推奨
        
        self.users[admin_email] = {
            'email': admin_email,
            'password_hash': self._hash_password(admin_password),
            'name': '管理者',
            'role': 'admin',
            'created_at': datetime.now().isoformat(),
            'profile': {
                'organization': 'PA Analyzer 運営',
                'location': '',
                'bio': 'システム管理者'
            }
        }
        self.save()
    
    def _hash_password(self, password):
        """パスワードをハッシュ化（SHA-256 + Salt）"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return f"{salt}:{pwd_hash}"
    
    def _verify_password(self, password, stored_hash):
        """パスワード検証"""
        try:
            salt, pwd_hash = stored_hash.split(':')
            test_hash = hashlib.sha256((password + salt).encode()).hexdigest()
            return test_hash == pwd_hash
        except:
            return False
    
    def register_user(self, email, password, name, organization='', location=''):
        """新規ユーザー登録"""
        
        # メールアドレス重複チェック
        if email in self.users:
            return False, "このメールアドレスは既に登録されています"
        
        # ユーザー作成
        self.users[email] = {
            'email': email,
            'password_hash': self._hash_password(password),
            'name': name,
            'role': 'user',  # 一般ユーザー
            'created_at': datetime.now().isoformat(),
            'last_login': None,
            'profile': {
                'organization': organization,
                'location': location,
                'bio': ''
            },
            'stats': {
                'total_analyses': 0,
                'last_analysis_date': None
            }
        }
        
        self.save()
        return True, "登録が完了しました"
    
    def authenticate(self, email, password):
        """認証"""
        
        if email not in self.users:
            return False, "メールアドレスまたはパスワードが正しくありません"
        
        user = self.users[email]
        
        if not self._verify_password(password, user['password_hash']):
            return False, "メールアドレスまたはパスワードが正しくありません"
        
        # ログイン成功 - 最終ログイン更新
        self.users[email]['last_login'] = datetime.now().isoformat()
        self.save()
        
        return True, user
    
    def get_user(self, email):
        """ユーザー情報取得"""
        return self.users.get(email)
    
    def update_user_stats(self, email):
        """ユーザー統計更新（解析実行時）"""
        if email in self.users:
            self.users[email]['stats']['total_analyses'] += 1
            self.users[email]['stats']['last_analysis_date'] = datetime.now().isoformat()
            self.save()
    
    def get_all_users(self):
        """全ユーザー取得（管理者用）"""
        return list(self.users.values())
    
    def update_profile(self, email, profile_data):
        """プロフィール更新"""
        if email in self.users:
            self.users[email]['profile'].update(profile_data)
            self.save()
            return True
        return False
    
    def change_password(self, email, old_password, new_password):
        """パスワード変更"""
        if email not in self.users:
            return False, "ユーザーが見つかりません"
        
        user = self.users[email]
        
        # 旧パスワード確認
        if not self._verify_password(old_password, user['password_hash']):
            return False, "現在のパスワードが正しくありません"
        
        # 新パスワード設定
        self.users[email]['password_hash'] = self._hash_password(new_password)
        self.save()
        
        return True, "パスワードを変更しました"


class UserAudioDatabase:
    """ユーザー別音源データベース"""
    
    def __init__(self, db_path='user_audio_data'):
        self.db_dir = Path(db_path)
        self.db_dir.mkdir(exist_ok=True)
    
    def _get_user_db_path(self, email):
        """ユーザー別DBファイルパス"""
        # メールアドレスをファイル名として使用（安全にエンコード）
        safe_email = email.replace('@', '_at_').replace('.', '_')
        return self.db_dir / f"{safe_email}.json"
    
    def add_analysis(self, email, analysis_data, metadata):
        """解析データ追加"""
        
        db_path = self._get_user_db_path(email)
        
        # 既存データ読み込み
        if db_path.exists():
            with open(db_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {'analyses': []}
        
        # 新規エントリ追加
        entry = {
            'id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'analysis': analysis_data
        }
        
        data['analyses'].append(entry)
        
        # 保存
        with open(db_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        return entry['id']
    
    def get_user_analyses(self, email, limit=None):
        """ユーザーの解析データ取得"""
        
        db_path = self._get_user_db_path(email)
        
        if not db_path.exists():
            return []
        
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        analyses = data.get('analyses', [])
        
        # 新しい順にソート
        analyses.sort(key=lambda x: x['timestamp'], reverse=True)
        
        if limit:
            return analyses[:limit]
        
        return analyses
    
    def delete_analysis(self, email, analysis_id):
        """解析データ削除"""
        
        db_path = self._get_user_db_path(email)
        
        if not db_path.exists():
            return False
        
        with open(db_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 該当データを削除
        original_count = len(data['analyses'])
        data['analyses'] = [a for a in data['analyses'] if a['id'] != analysis_id]
        
        if len(data['analyses']) < original_count:
            with open(db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        
        return False
    
    def get_all_analyses_for_admin(self):
        """全ユーザーの解析データ取得（管理者用）"""
        
        all_analyses = []
        
        for db_file in self.db_dir.glob('*.json'):
            try:
                with open(db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # メールアドレス復元
                email = db_file.stem.replace('_at_', '@').replace('_', '.')
                
                for analysis in data.get('analyses', []):
                    analysis['user_email'] = email
                    all_analyses.append(analysis)
            except:
                continue
        
        # 新しい順にソート
        all_analyses.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return all_analyses


def init_session_state():
    """セッションステート初期化"""
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if 'page' not in st.session_state:
        st.session_state.page = 'login'


def show_login_page(user_db):
    """ログインページ"""
    
    st.markdown("## 🔐 ログイン")
    
    with st.form("login_form"):
        email = st.text_input("メールアドレス", placeholder="example@email.com")
        password = st.text_input("パスワード", type="password")
        
        col1, col2 = st.columns(2)
        
        with col1:
            login_button = st.form_submit_button("ログイン", use_container_width=True)
        
        with col2:
            if st.form_submit_button("新規登録", use_container_width=True):
                st.session_state.page = 'register'
                st.rerun()
    
    if login_button:
        if email and password:
            success, result = user_db.authenticate(email, password)
            
            if success:
                st.session_state.authenticated = True
                st.session_state.user = result
                st.success(f"ようこそ、{result['name']}さん！")
                st.rerun()
            else:
                st.error(result)
        else:
            st.warning("メールアドレスとパスワードを入力してください")


def show_register_page(user_db):
    """新規登録ページ"""
    
    st.markdown("## 📝 新規ユーザー登録")
    
    with st.form("register_form"):
        st.markdown("### 基本情報")
        
        email = st.text_input("メールアドレス *", placeholder="example@email.com")
        password = st.text_input("パスワード *", type="password")
        password_confirm = st.text_input("パスワード（確認） *", type="password")
        
        st.markdown("### プロフィール")
        
        name = st.text_input("お名前 *", placeholder="山田太郎")
        organization = st.text_input("所属・団体", placeholder="例: フリーランス、〇〇スタジオ")
        location = st.text_input("活動地域", placeholder="例: 東京、大阪")
        
        st.caption("* は必須項目です")
        
        col1, col2 = st.columns(2)
        
        with col1:
            register_button = st.form_submit_button("登録", type="primary", use_container_width=True)
        
        with col2:
            if st.form_submit_button("ログインに戻る", use_container_width=True):
                st.session_state.page = 'login'
                st.rerun()
    
    if register_button:
        # バリデーション
        if not email or not password or not name:
            st.error("必須項目を入力してください")
            return
        
        if password != password_confirm:
            st.error("パスワードが一致しません")
            return
        
        if len(password) < 6:
            st.error("パスワードは6文字以上にしてください")
            return
        
        # 登録実行
        success, message = user_db.register_user(
            email, password, name, organization, location
        )
        
        if success:
            st.success(message)
            st.info("登録したメールアドレスとパスワードでログインしてください")
            
            if st.button("ログインページへ"):
                st.session_state.page = 'login'
                st.rerun()
        else:
            st.error(message)


def show_user_profile(user_db):
    """ユーザープロフィールページ"""
    
    user = st.session_state.user
    
    st.markdown("## 👤 プロフィール")
    
    with st.form("profile_form"):
        st.markdown("### 基本情報")
        st.text_input("メールアドレス", value=user['email'], disabled=True)
        st.text_input("登録日", value=datetime.fromisoformat(user['created_at']).strftime('%Y年%m月%d日'), disabled=True)
        
        if user.get('last_login'):
            st.text_input("最終ログイン", value=datetime.fromisoformat(user['last_login']).strftime('%Y年%m月%d日 %H:%M'), disabled=True)
        
        st.markdown("### プロフィール編集")
        
        name = st.text_input("お名前", value=user['name'])
        organization = st.text_input("所属・団体", value=user['profile'].get('organization', ''))
        location = st.text_input("活動地域", value=user['profile'].get('location', ''))
        bio = st.text_area("自己紹介", value=user['profile'].get('bio', ''), height=100)
        
        if st.form_submit_button("更新", type="primary"):
            user_db.users[user['email']]['name'] = name
            user_db.update_profile(user['email'], {
                'organization': organization,
                'location': location,
                'bio': bio
            })
            
            # セッションステート更新
            st.session_state.user = user_db.get_user(user['email'])
            
            st.success("プロフィールを更新しました")
            st.rerun()
    
    # パスワード変更
    st.markdown("---")
    st.markdown("### 🔒 パスワード変更")
    
    with st.form("password_form"):
        old_password = st.text_input("現在のパスワード", type="password")
        new_password = st.text_input("新しいパスワード", type="password")
        new_password_confirm = st.text_input("新しいパスワード（確認）", type="password")
        
        if st.form_submit_button("パスワード変更"):
            if not old_password or not new_password:
                st.error("全ての項目を入力してください")
            elif new_password != new_password_confirm:
                st.error("新しいパスワードが一致しません")
            elif len(new_password) < 6:
                st.error("パスワードは6文字以上にしてください")
            else:
                success, message = user_db.change_password(
                    user['email'], old_password, new_password
                )
                
                if success:
                    st.success(message)
                else:
                    st.error(message)
    
    # 統計情報
    st.markdown("---")
    st.markdown("### 📊 利用統計")
    
    stats = user.get('stats', {})
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("総解析数", stats.get('total_analyses', 0))
    
    with col2:
        last_analysis = stats.get('last_analysis_date')
        if last_analysis:
            st.metric("最終解析日", datetime.fromisoformat(last_analysis).strftime('%Y/%m/%d'))
        else:
            st.metric("最終解析日", "未実施")


def show_admin_dashboard(user_db, audio_db):
    """管理者ダッシュボード"""
    
    st.markdown("## 🛡️ 管理者ダッシュボード")
    
    tab1, tab2, tab3 = st.tabs(["📊 統計", "👥 ユーザー管理", "🎵 音源管理"])
    
    with tab1:
        show_admin_stats(user_db, audio_db)
    
    with tab2:
        show_admin_users(user_db)
    
    with tab3:
        show_admin_audio(audio_db)


def show_admin_stats(user_db, audio_db):
    """管理者統計ページ"""
    
    st.markdown("### 📊 システム統計")
    
    users = user_db.get_all_users()
    all_analyses = audio_db.get_all_analyses_for_admin()
    
    # サマリーメトリクス
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("総ユーザー数", len(users))
    
    with col2:
        st.metric("総解析数", len(all_analyses))
    
    with col3:
        active_users = len([u for u in users if u.get('stats', {}).get('total_analyses', 0) > 0])
        st.metric("アクティブユーザー", active_users)
    
    with col4:
        if all_analyses:
            avg_per_user = len(all_analyses) / max(len(users), 1)
            st.metric("ユーザーあたり平均解析数", f"{avg_per_user:.1f}")
        else:
            st.metric("ユーザーあたり平均解析数", "0.0")
    
    st.markdown("---")
    
    # 最近のアクティビティ
    st.markdown("### 🕐 最近のアクティビティ")
    
    if all_analyses:
        recent = all_analyses[:10]
        
        for analysis in recent:
            timestamp = datetime.fromisoformat(analysis['timestamp'])
            name = analysis['metadata'].get('analysis_name', '名称未設定')
            user_email = analysis.get('user_email', '不明')
            
            st.markdown(f"""
            **{timestamp.strftime('%Y/%m/%d %H:%M')}** - {user_email}  
            📝 {name}
            """)
            st.markdown("---")
    else:
        st.info("まだ解析データがありません")


def show_admin_users(user_db):
    """管理者ユーザー管理ページ"""
    
    st.markdown("### 👥 ユーザー一覧")
    
    users = user_db.get_all_users()
    
    # 検索・フィルター
    search = st.text_input("🔍 検索", placeholder="メールアドレス、名前、所属で検索")
    
    # フィルタリング
    if search:
        filtered_users = [
            u for u in users
            if search.lower() in u['email'].lower()
            or search.lower() in u['name'].lower()
            or search.lower() in u['profile'].get('organization', '').lower()
        ]
    else:
        filtered_users = users
    
    # 統計でソート
    filtered_users.sort(
        key=lambda u: u.get('stats', {}).get('total_analyses', 0),
        reverse=True
    )
    
    st.write(f"**表示: {len(filtered_users)}人 / 全{len(users)}人**")
    
    # ユーザー一覧表示
    for user in filtered_users:
        with st.expander(
            f"{'🛡️ ' if user['role'] == 'admin' else '👤 '}{user['name']} ({user['email']})",
            expanded=False
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**基本情報**")
                st.write(f"**メール**: {user['email']}")
                st.write(f"**名前**: {user['name']}")
                st.write(f"**権限**: {user['role']}")
                st.write(f"**登録日**: {datetime.fromisoformat(user['created_at']).strftime('%Y/%m/%d')}")
                
                if user.get('last_login'):
                    st.write(f"**最終ログイン**: {datetime.fromisoformat(user['last_login']).strftime('%Y/%m/%d %H:%M')}")
            
            with col2:
                st.markdown("**プロフィール**")
                st.write(f"**所属**: {user['profile'].get('organization', '未設定')}")
                st.write(f"**地域**: {user['profile'].get('location', '未設定')}")
                
                if user['profile'].get('bio'):
                    st.write(f"**自己紹介**: {user['profile']['bio']}")
                
                st.markdown("**利用統計**")
                stats = user.get('stats', {})
                st.write(f"**総解析数**: {stats.get('total_analyses', 0)}")
                
                if stats.get('last_analysis_date'):
                    st.write(f"**最終解析**: {datetime.fromisoformat(stats['last_analysis_date']).strftime('%Y/%m/%d')}")


def show_admin_audio(audio_db):
    """管理者音源管理ページ"""
    
    st.markdown("### 🎵 アップロード音源一覧")
    
    all_analyses = audio_db.get_all_analyses_for_admin()
    
    if not all_analyses:
        st.info("まだアップロードされた音源がありません")
        return
    
    # フィルター
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_user = st.text_input("ユーザーで検索", placeholder="メールアドレス")
    
    with col2:
        search_name = st.text_input("音源名で検索", placeholder="解析名")
    
    with col3:
        search_venue = st.text_input("会場で検索", placeholder="会場名")
    
    # フィルタリング
    filtered = all_analyses
    
    if search_user:
        filtered = [a for a in filtered if search_user.lower() in a.get('user_email', '').lower()]
    
    if search_name:
        filtered = [a for a in filtered if search_name.lower() in a['metadata'].get('analysis_name', '').lower()]
    
    if search_venue:
        filtered = [a for a in filtered if search_venue.lower() in a['metadata'].get('venue', '').lower()]
    
    st.write(f"**表示: {len(filtered)}件 / 全{len(all_analyses)}件**")
    
    # 音源一覧表示
    for analysis in filtered:
        timestamp = datetime.fromisoformat(analysis['timestamp'])
        name = analysis['metadata'].get('analysis_name', '名称未設定')
        user_email = analysis.get('user_email', '不明')
        venue = analysis['metadata'].get('venue', '不明')
        
        with st.expander(
            f"🎵 {name} - {user_email} ({timestamp.strftime('%Y/%m/%d %H:%M')})",
            expanded=False
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📅 基本情報**")
                st.write(f"**アップロード日時**: {timestamp.strftime('%Y年%m月%d日 %H:%M')}")
                st.write(f"**ユーザー**: {user_email}")
                st.write(f"**解析名**: {name}")
                st.write(f"**ID**: {analysis['id']}")
            
            with col2:
                st.markdown("**🏛️ 会場・機材情報**")
                st.write(f"**会場**: {venue}")
                st.write(f"**キャパ**: {analysis['metadata'].get('venue_capacity', '不明')}人")
                st.write(f"**ミキサー**: {analysis['metadata'].get('mixer', '不明')}")
                st.write(f"**PA**: {analysis['metadata'].get('pa_system', '不明')}")
                st.write(f"**バンド編成**: {analysis['metadata'].get('band_lineup', '不明')}")
            
            # 解析結果サマリー
            st.markdown("---")
            st.markdown("**📊 解析結果サマリー**")
            
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
