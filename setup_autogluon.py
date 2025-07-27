#!/usr/bin/env python3
"""
AutoGluon 1.0 Setup and Environment Check
AutoGluon 1.0環境セットアップ・確認スクリプト

機能:
- AutoGluon 1.0依存関係インストール
- A100 GPU確認
- データファイル確認
- システム情報表示
"""

import subprocess
import sys
import os
import platform
import psutil

def run_command(command, description):
    """
    コマンド実行ヘルパー
    """
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description}完了")
            return True
        else:
            print(f"❌ {description}失敗: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description}エラー: {e}")
        return False

def check_system_info():
    """
    システム情報確認
    """
    print("💻 システム情報:")
    print(f"   🐍 Python: {sys.version}")
    print(f"   💾 OS: {platform.system()} {platform.release()}")
    print(f"   🖥️  CPU: {psutil.cpu_count()}コア")
    print(f"   🧠 RAM: {psutil.virtual_memory().total // (1024**3)}GB")

def check_gpu():
    """
    GPU確認
    """
    print("\n🔥 GPU確認:")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"   ✅ GPU検出: {gpu_count}基")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory // (1024**3)
                print(f"   🚀 GPU {i}: {gpu_name} ({gpu_memory}GB)")
                
            # A100確認
            if "A100" in gpu_name:
                print("   🎯 A100検出! 最高性能で実行可能")
            return True
        else:
            print("   ⚠️  GPU未検出、CPU実行となります")
            return False
    except ImportError:
        print("   ❌ PyTorch未インストール")
        return False

def install_dependencies():
    """
    依存関係インストール
    """
    print("\n📦 AutoGluon 1.0依存関係インストール:")
    
    # requirements_autogluon.txtから読み込み
    if os.path.exists('requirements_autogluon.txt'):
        success = run_command(
            f"{sys.executable} -m pip install -r requirements_autogluon.txt",
            "AutoGluon 1.0インストール"
        )
        if not success:
            print("❌ インストール失敗")
            return False
    else:
        print("❌ requirements_autogluon.txt未発見")
        # 直接インストール
        packages = [
            "autogluon>=1.0.0",
            "pandas>=2.0",
            "numpy>=1.21",
            "scikit-learn>=1.3",
            "torch>=2.0",
            "psutil>=5.8.0",
            "tqdm>=4.64.0"
        ]
        
        for package in packages:
            success = run_command(
                f"{sys.executable} -m pip install '{package}'",
                f"{package}インストール"
            )
            if not success:
                print(f"❌ {package}インストール失敗")
                return False
    
    return True

def verify_autogluon():
    """
    AutoGluon 1.0動作確認
    """
    print("\n🧪 AutoGluon 1.0動作確認:")
    try:
        from autogluon.tabular import TabularPredictor
        import autogluon
        
        version = autogluon.__version__
        print(f"   ✅ AutoGluon {version}インストール済み")
        
        # バージョン確認
        if version.startswith('1.'):
            print("   🎉 AutoGluon 1.0系確認!")
            return True
        else:
            print(f"   ⚠️  AutoGluon {version}検出、1.0系推奨")
            return False
            
    except ImportError as e:
        print(f"   ❌ AutoGluonインポートエラー: {e}")
        return False

def check_data_files():
    """
    データファイル確認
    """
    print("\n📂 データファイル確認:")
    
    required_files = [
        'data/train.tsv',
        'data/test.tsv'
    ]
    
    all_present = True
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            print(f"   ✅ {file_path} ({size_mb:.1f}MB)")
        else:
            print(f"   ❌ {file_path} 未発見")
            all_present = False
    
    return all_present

def main():
    """
    メインセットアップ処理
    """
    print("=" * 60)
    print("🚀 AutoGluon 1.0 Setup & Environment Check")
    print("📊 OpenML Benchmark 1位システム環境構築")
    print("=" * 60)
    
    # システム情報確認
    check_system_info()
    
    # GPU確認
    gpu_available = check_gpu()
    
    # 依存関係インストール
    deps_success = install_dependencies()
    
    if deps_success:
        # AutoGluon動作確認
        autogluon_ok = verify_autogluon()
        
        # データファイル確認  
        data_ok = check_data_files()
        
        print("\n" + "=" * 60)
        print("📋 セットアップ結果:")
        print(f"   🔧 依存関係: {'✅' if deps_success else '❌'}")
        print(f"   🐍 AutoGluon 1.0: {'✅' if autogluon_ok else '❌'}")
        print(f"   🔥 GPU: {'✅' if gpu_available else '⚠️ CPU'}")
        print(f"   📂 データ: {'✅' if data_ok else '❌'}")
        
        if all([deps_success, autogluon_ok, data_ok]):
            print("\n🎉 セットアップ完了!")
            print("💡 実行準備完了:")
            print("   python autogluon_1_0_implementation.py")
        else:
            print("\n⚠️  セットアップに問題があります")
            if not data_ok:
                print("💡 データファイルをdata/フォルダに配置してください")
    else:
        print("\n❌ セットアップ失敗")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 