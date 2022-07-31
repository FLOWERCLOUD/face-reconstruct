# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from math import pi
def test():
    N = 32
    bottom = 0
    max_height = 10

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    theta = np.concatenate((theta,np.array([0])))
    radii = max_height*np.random.rand(N)
    radii = np.concatenate((radii,np.array([30])))
    width = (2*np.pi) / N*0.8


    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, radii, width=width, bottom=bottom)
    ax.set_theta_offset(pi/2)
    ax.set_theta_direction(-1)
    # Use custom colors and opacity
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(r / 10.))
        #bar.set_facecolor([0,0,0,0])
        bar.set_alpha(0.8)
    bars[len(bars)-1].set_alpha(0.0)
    plt.savefig('E:\workspace\dataset\hairstyles\hair\convert_hair_dir/plot_bin/foo.png')
    #plt.show()
def generate_plot_fig(value,output_path):
    N = value.shape[0]

    bottom = 0
    max_height = 10

    theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
    theta = np.concatenate((theta,np.array([0])))
    #radii = max_height*np.random.rand(N)
    radii = value
    radii = np.concatenate((radii,np.array([400])))
    width = (2*np.pi) / N


    ax = plt.subplot(111, polar=True)
    bars = ax.bar(theta, radii, width=width, bottom=bottom)
    ax.set_theta_offset(pi/2)
    ax.set_theta_direction(-1)
    # Use custom colors and opacity
    all_count = value.shape[0]
    count = 1
    for r, bar in zip(radii, bars):
        bar.set_facecolor(plt.cm.jet(10*count))
        #bar.set_facecolor([0,0,0,0])
        bar.set_alpha(0.8)
        count+=1
    bars[len(bars)-1].set_alpha(0.0)
    plt.savefig(output_path)
    plt.close('all') #需关掉这个，不然下次画图有问题
if __name__ == '__main__':
    from fitting.util import load_binary_pickle
    hairs_seg_bin = load_binary_pickle('E:/workspace/dataset/hairstyles/hair/convert_hair_dir/seg_bin/seg_bin.pkl')
    out_dir = 'E:\workspace\dataset\hairstyles\hair\convert_hair_dir/plot_bin/'
    for key, value in hairs_seg_bin.items():
        generate_plot_fig(value,out_dir+key+'.png')
        #break