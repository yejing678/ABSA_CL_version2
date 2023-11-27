import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5), dpi=100)
shift_stage_epoch = ['1', '2', '3', '4', '5']
laptop_accuracy = [77.745, 75.94, 78.53, 77.51, 79.7]
laptop_f1 = [73.085, 71.145, 75.01, 72.635, 75.35]
restaurant_acc = [84.335, 84.42, 84.51, 83.93, 85.3]
restaurant_f1 = [76.03, 76.615, 76.36, 75.855, 78.25]
plt.plot(shift_stage_epoch, laptop_accuracy, c='black', label="laptop acc")
plt.plot(shift_stage_epoch, laptop_f1, c='black', linestyle='--', label="laptop f1")
plt.plot(shift_stage_epoch, restaurant_acc, c='black', ls='-.', label="restaurant acc")
plt.plot(shift_stage_epoch, restaurant_f1, c='black', linestyle=':', label="restaurant f1")
plt.scatter(shift_stage_epoch, laptop_accuracy, c='black', marker='o')
plt.scatter(shift_stage_epoch, laptop_f1, c='black', marker='+')
plt.scatter(shift_stage_epoch, restaurant_acc, c='black', marker='^')
plt.scatter(shift_stage_epoch, restaurant_f1, c='black', marker='*')
plt.legend(loc='best')
plt.yticks(range(65, 90, 5))
plt.grid(True, linestyle='--', alpha=0.5)
plt.xlabel("shift stage epoch", fontdict={'size': 16})

plt.savefig("./plot.png")  # 保存图像

plt.show()
