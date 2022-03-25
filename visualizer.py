import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DataAnalyzer:
    def __init__(self, input_location):
        self.df = pd.read_csv(input_location)
        self.avg_fps = 0
        self.avg_position = (0,0)

    def export_to_csv(self, output_location):
        self.df.to_csv(output_location)

    def add_fps_to_df(self):
        timestamps = self.df['t'].to_numpy()
        fps = [0]

        for i in range(1, len(timestamps)):
            fps.append(1/(timestamps[i] - timestamps[i-1])/1000)

        self.df.insert(len(self.df.columns), 'fps', fps)
        
    def visualize_fps_raw(self):
        if 'fps' not in self.df.columns:
            self.add_fps_to_df()
            
        fps = self.df['fps'].to_numpy()
        timesteps = np.linspace(0, fps.size - 1, fps.size)
        fig = plt.figure()
        ax = fig.add_subplot()
        plt.scatter(timesteps, fps, c='blue')
        ax.set_xlabel('frames')
        ax.set_ylabel('fps')
        plt.show()



    def visualize3D(self, input_location):
        x_vec = self.df['x'].to_numpy()
        y_vec = self.df['y'].to_numpy()
        z_vec = self.df['z'].to_numpy()
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        
        for i in range(0, len(x_vec)):
            ax.scatter(x_vec[i], y_vec[i], z_vec[i], c='blue')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        
        # Assume capture dimenstions are 1920x1080
        ax.set_xlim((0,1920))
        ax.set_ylim((0,1080))
        
        plt.show()


    def visualize2D(self):
        x_vec = self.df['x'].to_numpy()
        y_vec = self.df['y'].to_numpy()
        
        fig = plt.figure()
        ax = fig.add_subplot()
        
        for i in range(0, len(x_vec)):
            ax.scatter(x_vec[i], y_vec[i], c='blue')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
        # Assume capture dimenstions are 1920x1080
        ax.set_xlim((0,1920))
        ax.set_ylim((0,1080))
        
        plt.show()


if __name__=='__main__':
    vis = DataAnalyzer('logs/records_bottle_0.csv')
    vis.add_fps_to_df()
    #vis.visualize_fps_raw()
    vis.visualize2D()
    
