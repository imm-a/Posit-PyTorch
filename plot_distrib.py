import matplotlib.pyplot as plt

def plot_pos(all_fp,all_posit,n,es):
    str1 = "Posit("+str(n)+","+str(es)+")"
    str2 = str(n)+"_"+str(es)+".png"
    plt.rcParams.update({'figure.figsize':(7,5), 'figure.dpi':200})
    plt.figure()
    plt.hist(all_fp[0], alpha = 0.5, bins = 50, color='b', label = 'FP')
    plt.hist(all_posit[0],alpha = 0.5,  bins = 50, color='r',label = str1)
    plt.savefig(str2)
