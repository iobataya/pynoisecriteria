from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

class NC_table:
    """
    NC curve and estimation methods
    Call input_levels() method to input noise level at each frequency
    L.L.Beranek, J. Acoust. Soc. Amer. 25, 313-321 (1953)
    
    """
    def __init__(self):
        """
        Initialize NC curve
        """
        self.octave_bands = np.array([63, 125, 250, 500, 1000, 2000, 4000, 8000])
        self.level_mat = np.array([
            [47, 36, 29, 22, 17, 14, 12, 11],
            [51, 40, 33, 26, 22, 19, 17, 16],
            [54, 44, 37, 31, 27, 24, 22, 21],
            [57, 48, 41, 35, 31, 29, 28, 27],
            [60, 52, 45, 40, 36, 34, 33, 32],
            [64, 56, 50, 45, 41, 39, 38, 37],
            [67, 60, 54, 49, 46, 44, 43, 42],
            [71, 64, 58, 54, 51, 49, 48, 47],
            [74, 67, 62, 58, 56, 54, 53, 52],
            [77, 71, 67, 63, 61, 59, 58, 57],
        ], dtype=float)
        self.levels = ["NC-"+str(i) for i in range(15,65,5)]
        self.data = None
    
    def input_levels(self):
        """
        Estimate NC level by noise level at octave-bands by interactive way.
        """
        input_data = []
        print("Input noise level.")
        # Input noise levels from user
        for band in self.octave_bands:
            l = float(input(f"Level at {band}Hz: "))
            if l >= 0:
                input_data.append(l)
        self.data = np.array(input_data,dtype=float)
        
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-NC-criteria"
        
        self.save(filename + '.csv')

        self.calculate()
        report = self.generate_report()        
        print(report)
        self.hr()
        print(f"NC level: {self.nc_level}")
        # Print frequencies of maximum level
        for f in self.freqs:
            print(f"  Maximum level at {self.octave_bands[f]} Hz")
        self.hr()
        self.plot_mat(overlay_ar=self.data, filename=filename+'.png')
        

    def load(self, filename):
        loaddata = np.loadtxt(filename, delimiter=",", skiprows=1)
        self.data = loaddata.T[1]

    def save(self, filename):
        if self.data is None:
            raise ValueError("No data")
        x_arr = self.octave_bands
        y_arr = self.data
        savedata = np.column_stack((x_arr, y_arr))
        np.savetxt(filename, savedata, delimiter=",", header="freq Hz,Noise dB", comments='', fmt="%.2f")

    def calculate(self):
        if self.data is None:
            raise ValueError("No input data.")
        # Set boolean matrix of comparison of data and NC curve at each frequency
        self.lt_NC = (self.data < self.level_mat)  # Boolean matrix of the condition Data < NC plot
        self.gt_NC = (self.data >= self.level_mat)  # Boolean matrix of the condition Data >= NC plot

        # Find NC level
        self.all_lt_NC_rows = np.where(np.all(self.lt_NC, axis=1))[0]  # Row indecies where all True in a row
        min_NC_row = self.all_lt_NC_rows.min()  # Minimum row index
        self.nc_level = self.levels[min_NC_row]  # Name of the maximum noise level

        # Find frequencies of maximum noise
        self.gt_NC_idx = np.where(self.gt_NC)
        max_row = self.gt_NC_idx[0].max()
        self.freq_at_max_level = np.where(self.gt_NC_idx[0] == max_row)[0]
        self.freqs = self.gt_NC_idx[1][self.freq_at_max_level]
    
    def hr(self):
        print('-' * 72)

    def generate_report(self) -> str:
        # print column headers
        print(f"{'':>7}", end="")
        for band in self.octave_bands:
            print(f"{band:>6} ", end="")
        print()
        
        # print data
        if self.data is not None:
            print(f"{'Data':>7}", end="")
            for data in self.data:
                print(f"{data:>6} ", end="")
            print()     

        # print rows with labels
        mat = self.level_mat
        flag_mat = self.gt_NC
        for i, level in enumerate(self.levels):
            print(f"{level:>7}", end="")
            for j, value in enumerate(mat[i]):
                if flag_mat is None:
                    print(f"{value:7.1f} ", end="")
                else:
                    if flag_mat[i][j]==1:
                        f = '*'
                    else:
                        f = ' '
                    print(f"{value:6.1f}{f}", end="")                        
            print()
    
    def plot_mat(self, overlay_ar=None, filename=None):
        # グラフのプロット
        x_arr = self.octave_bands
        y_mat = self.level_mat
        legends = self.levels
        for i in range(y_mat.shape[0]):
            if legends is None:
                plt.plot(x_arr[:y_mat.shape[1]], y_mat[i], label=f"Line {i}")
            else:
                plt.plot(x_arr[:y_mat.shape[1]], y_mat[i], label=legends[i])

        # 追加プロットがあれば描画
        if overlay_ar is not None:
            plt.plot(x_arr[:y_mat.shape[1]], overlay_ar, marker='s', label="Data")

        # X軸を対数軸に設定
        plt.xscale('log')

        # グラフのラベル設定
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")
        plt.title("NC curves")

        # 凡例の表示
        #plt.legend()
        # ラベルをプロットの右側に手動で配置
        for i in range(y_mat.shape[0]):
            if legends is None:
                plt.text(x_arr[-1] * 1.2, y_mat[i, -1], f"Line {i + 1}", va='center')
            else:
                plt.text(x_arr[-1] * 1.2, y_mat[i, -1], legends[i], va='center')            

        # グラフの表示・保存
        if filename is not None:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
        else:
            plt.show()

if __name__ == "__main__":
    nc = NC_table()
    nc.input_levels()