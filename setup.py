#!/usr/bin/env python3
"""
PA Audio Analyzer - クイックスタートスクリプト

このスクリプトは初期セットアップを支援します。
"""

import os
import json
from pathlib import Path


def create_directory_structure():
    """必要なディレクトリを作成"""
    print("📁 ディレクトリ構造を作成中...")
    
    dirs = [
        'user_audio_data',
        'logs',
        'backups'
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"  ✓ {dir_name}/")
    
    print()


def check_files():
    """必要なファイルの存在確認"""
    print("📝 必要なファイルを確認中...")
    
    required_files = [
        'pa_analyzer_v3_final.py',
        'auth_system.py',
        'pa_analyzer_with_auth.py',
        'requirements.txt'
    ]
    
    missing = []
    
    for filename in required_files:
        if Path(filename).exists():
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ {filename} が見つかりません")
            missing.append(filename)
    
    print()
    
    if missing:
        print("⚠️ 以下のファイルが不足しています:")
        for f in missing:
            print(f"  - {f}")
        print()
        return False
    
    return True


def create_config():
    """設定ファイルを作成"""
    print("⚙️ 設定ファイルを作成中...")
    
    config = {
        'app_name': 'PA Audio Analyzer',
        'version': '3.0',
        'default_admin': {
            'email': 'admin@pa-analyzer.local',
            'password': 'admin123'
        },
        'security': {
            'min_password_length': 6,
            'session_timeout': 3600
        },
        'features': {
            'user_registration': True,
            'admin_dashboard': True,
            'data_export': False
        }
    }
    
    config_path = Path('config.json')
    
    if not config_path.exists():
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        print("  ✓ config.json を作成しました")
    else:
        print("  • config.json は既に存在します")
    
    print()


def show_next_steps():
    """次のステップを表示"""
    print("=" * 60)
    print("✅ セットアップ完了！")
    print("=" * 60)
    print()
    print("🚀 次のステップ:")
    print()
    print("1. アプリを起動:")
    print("   streamlit run pa_analyzer_with_auth.py")
    print()
    print("2. ブラウザで開く:")
    print("   http://localhost:8501")
    print()
    print("3. デフォルト管理者でログイン:")
    print("   メール: admin@pa-analyzer.local")
    print("   パスワード: admin123")
    print()
    print("⚠️ セキュリティ:")
    print("   初回ログイン後、必ずパスワードを変更してください")
    print()
    print("📖 詳細:")
    print("   AUTH_INTEGRATION_GUIDE.md を参照")
    print()


def main():
    """メイン処理"""
    print()
    print("=" * 60)
    print("PA Audio Analyzer - セットアップ")
    print("=" * 60)
    print()
    
    # ディレクトリ作成
    create_directory_structure()
    
    # ファイル確認
    if not check_files():
        print("❌ セットアップを完了できません")
        print("   不足しているファイルをダウンロードしてください")
        return
    
    # 設定ファイル作成
    create_config()
    
    # 次のステップ表示
    show_next_steps()


if __name__ == "__main__":
    main()
