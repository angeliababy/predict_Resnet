from keras.models import load_model
# from matplotlib.font_manager import FontProperties
import cv2
import numpy as np
import exptBikeNYC
size =10

model = exptBikeNYC.build_model(False)
model.load_weights('MODEL/c3.p3.t3.resunit4.lr0.0002.best.h5')

f = open("area.csv", "r")
# 临时存储某时间的人数
person_num = []
# 存储各时间的人数尺寸(n,3,3)
imgs = []

i, l = 0, 0
for line in f:
    l += 1
    if l == 1:
        continue
    i += 1
    line = line.strip().split(',')
    # 将人数转化为小于1的数，后面求实际人数需转化过来
    number = (float(line[2]) - 0) / (3073 - 0) * 2 - 1
    person_num.append(number)
    # 每次读16个数据
    if i % (128) == 0:
        # 转化成一维数组
        person_num = np.array(person_num)
        # 改变形状，类似图像形式
        person_num = person_num.reshape(16, 8)
        imgs.append(person_num)
        i = 0
        person_num = []

# 训练数据（输入三种类型的数据，并各自转化为多通道形式）
train_x1, train_x2, train_x3, train_y = [], [], [], []
for i in range(1300, 1305):
# 取短期、周期、趋势三组件数据，各不同长度序列
    image1 = [imgs[i - 3], imgs[i - 2], imgs[i - 1]]
    image2 = [imgs[i - 72], imgs[i - 48], imgs[i - 24]]
    image3 = [imgs[i - 484], imgs[i - 336], imgs[i - 168]]
    train_x1.append(image1)
    train_x2.append(image2)
    train_x3.append(image3)
    lab = [imgs[i]]
    train_y.append(lab)  # 最终输出
train_x = [np.array(train_x1), np.array(train_x2), np.array(train_x3)]
train_y = np.array(train_y)

# X_test, Y_test=exptBikeNYC.main()
predict_y = model.predict(train_x)
print((train_y+1)/2*3073)
print(((predict_y+1)/2*3073).astype(int))
# print((predict_y*(60923+192687)-192687))

