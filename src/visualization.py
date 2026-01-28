from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class PlotConfig:
    """플롯 스타일 설정을 위한 데이터 클래스"""
    label_size: int = 24
    label_weight: str = "bold"
    font_weight: str = 'bold'
    font_size: int = 24
    cmap: str = 'turbo'  # 컬러맵 이름 (문자열)
    title_size: int = 28
    title_weight: str = "bold"
    legend_fontsize: int = 14
    line_width: int = 3
    marker_size: int = 8
    zero_transparent: bool = True  # 0값 투명 처리 여부
    
    def apply_settings(self):
        """matplotlib에 설정 적용"""
        plt.rcParams['axes.labelsize'] = self.label_size
        plt.rcParams["axes.labelweight"] = self.label_weight
        plt.rcParams['font.weight'] = self.font_weight
        plt.rcParams['font.size'] = self.font_size
        plt.rcParams['image.cmap'] = self.cmap  # 컬러맵 이름만 설정
        plt.rcParams['axes.titlesize'] = self.title_size
        plt.rcParams["axes.titleweight"] = self.title_weight
        plt.rcParams['legend.fontsize'] = self.legend_fontsize
        plt.rcParams['lines.linewidth'] = self.line_width
        plt.rcParams['lines.markersize'] = self.marker_size