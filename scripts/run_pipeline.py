import subprocess
import sys
from pathlib import Path
# ==========================================
# 통합 시뮬레이션 & 분석 파이프라인 (Master)
# ==========================================

# 1. 스캔 범위 설정
# 부동소수점 오차 방지를 위해 값을 명확히 정의
SCAN_CONFIG = {
    "startA": round(10e-6, 12),
    "stopA": round(10e-5, 12),
    "stepA": round(1e-6, 12),
    "startD": round(1.0e-5, 12),
    "stopD": round(2.2e-5, 12),
    "stepD": round(0.2e-6, 12)
}

PYTHON_EXE = sys.executable  # 현재 가상환경의 파이썬 사용


def get_session_dir_name() -> str:
    """Build the simulation session directory name using the same format as scan_alphac_pyele.py."""
    a_range = f"A{SCAN_CONFIG['startA']:.2e}-{SCAN_CONFIG['stopA']:.2e}"
    d_range = f"D{SCAN_CONFIG['startD']:.2e}-{SCAN_CONFIG['stopD']:.2e}"
    return f"scan_{a_range}_{d_range}"

def run_simulation():
    """단계 1: Elegant 시뮬레이션을 실행하여 데이터 생성"""
    print("\n" + "="*60)
    print("STEP 1: Starting Simulation Sweep...")
    print("="*60)
    
    cmd = [
        PYTHON_EXE, "src/scan_alphac_pyele.py",
        "--startA", str(SCAN_CONFIG["startA"]),
        "--stopA", str(SCAN_CONFIG["stopA"]),
        "--stepA", str(SCAN_CONFIG["stepA"]),
        "--startD", str(SCAN_CONFIG["startD"]),
        "--stopD", str(SCAN_CONFIG["stopD"]),
        "--stepD", str(SCAN_CONFIG["stepD"])
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
