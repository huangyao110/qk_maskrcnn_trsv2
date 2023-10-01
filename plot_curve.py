import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_loss_and_lr(train_loss, learning_rate,name):
    try:
        x = list(range(len(train_loss)))
        fig, ax1 = plt.subplots(1, 1)
        ax1.plot(x, train_loss, 'r', label='loss')
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss")
        ax1.set_title("Train Loss and lr")
        plt.legend(loc='best')

        ax2 = ax1.twinx()
        ax2.plot(x, learning_rate, label='lr')
        ax2.set_ylabel("Learning Rate")
        ax2.set_xlim(0, len(train_loss))  # 设置横坐标整数间隔
        plt.legend(loc='best')

        handles1, labels1 = ax1.get_legend_handles_labels() # 获取第一个图例的句柄和标签
        handles2, labels2 = ax2.get_legend_handles_labels() # 获取第二个图例的句柄和标签
        plt.legend(handles1 + handles2, labels1 + labels2, loc='upper right')

        fig.subplots_adjust(right=0.8)  # 防止出现保存图片显示不全的情况
        fig.savefig('./loss_and_lr{0}_{1}.png'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), name))
        plt.close()
        print("successful save loss curve! ")
    except Exception as e:
        print(e)


def plot_map(mAP, name):
    try:
        x = list(range(len(mAP)))
        plt.plot(x, mAP, label='mAp')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')
        plt.title('Eval ,MAP')
        plt.xlim(0, len(mAP))
        plt.legend(loc='best')
        plt.savefig('./mAP_{0}.png'.format(name))
        plt.close()
        print("successful save mAP curve!")
    except Exception as e:
        print(e)






