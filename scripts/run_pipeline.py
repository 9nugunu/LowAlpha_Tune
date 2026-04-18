import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DEFAULT_SCAN_CONFIG

# ==========================================
# 통합 시뮬레이션 & 분석 파이프라인 (Master)
# ==========================================

PYTHON_EXE = sys.executable  # 현재 가상환경의 파이썬 사용


def get_session_dir_name() -> str:
    """Build the default simulation session directory name from central config."""
    return DEFAULT_SCAN_CONFIG.session_dir_name()

def run_simulation():
    """단계 1: Elegant 시뮬레이션을 실행하여 데이터 생성"""
    print("\n" + "="*60)
    print("STEP 1: Starting Simulation Sweep...")
    print("="*60)
    
    cmd = [
        PYTHON_EXE, "src/scan_alphac_pyele.py",
        "--startA", str(DEFAULT_SCAN_CONFIG.startA),
        "--stopA", str(DEFAULT_SCAN_CONFIG.stopA),
        "--stepA", str(DEFAULT_SCAN_CONFIG.stepA),
        "--startD", str(DEFAULT_SCAN_CONFIG.startD),
        "--stopD", str(DEFAULT_SCAN_CONFIG.stopD),
        "--stepD", str(DEFAULT_SCAN_CONFIG.stepD)
    ]
    
    # 프로세스 실행 및 출력 실시간 표시
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n[!] Simulation failed. Aborting pipeline.")
        sys.exit(1)

def run_analysis():
    """단계 2: 생성된 데이터를 분석하여 그리드 생성 및 컨투어 플롯 그리기"""
    print("\n" + "="*60)
    print("STEP 2: Starting Data Analysis & Plotting...")
    print("="*60)
    
    # 시뮬레이션과 동일한 세션 디렉터리를 명시적으로 넘겨 latest-lookup 의존성을 제거
    cmd = [PYTHON_EXE, "src/process_scan_results.py", "--dir", get_session_dir_name()]
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n[!] Analysis failed.")
        sys.exit(1)

def main():
    print("Beam Dynamics Pipeline Started.")
    
    # 1. 시뮬레이션 실행
    run_simulation()
    
    # 2. 분석 및 플로팅 실행
    run_analysis()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"Check 'output/process_scan_results/' for plots.")
    print("="*60)

if __name__ == "__main__":
    main()
