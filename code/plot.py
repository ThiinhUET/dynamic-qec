import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import StrMethodFormatter

with open("newnew.log", "r") as f:
    loss_file = f.readlines()
line_num = 0
items_num = 0
ave_reward = 0.0
for line in loss_file:
    if line[:5] != "Start":
        reward = float(line.strip().split(": ")[1])
        ave_reward = ave_reward * items_num / (items_num + 1) + reward * 1.0 / (items_num + 1)
        items_num += 1
    else:
        break
    line_num += 1

ave_reward_list = [ave_reward]
ave_loss_list = []
reward_list = []
loss_list = []
line_num += 1
items_num = 0
ave_reward = 0.0
ave_loss = 0.0
while loss_file[line_num][:8] != "Complete":
    line = loss_file[line_num]
    if line[:5] != "Start":
        reward = float(line.strip().split(":")[1])
        line_num += 1
        line = loss_file[line_num]
        loss = float(line.strip().split(":")[1])
        line_num += 1
        ave_reward = ave_reward * items_num / (items_num + 1) + reward * 1.0 / (items_num + 1)
        ave_loss = ave_loss * items_num / (items_num + 1) + loss * 1.0 / (items_num + 1)
        items_num += 1
        loss_list.append(loss)
        reward_list.append(reward)
    else:
        ave_reward_list.append(ave_reward)
        ave_loss_list.append(ave_loss)
        line_num += 1
        items_num = 0
        ave_reward = 0.0
        ave_loss = 0.0
xfmt = ScalarFormatter(useMathText=True)
xfmt.set_powerlimits((-3,3))
plt.figure(figsize=(5, 4))
plt.grid(color='0.90', linestyle='--', linewidth=0.2)
# plt.plot(loss_list)
plt.plot(ave_reward_list[1:])
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.gca().xaxis.set_major_formatter(xfmt)
plt.savefig("loss.pdf")
plt.show()

# # loss coeff 1
# plt.figure(figsize=(5, 4))
# p = [0.0109547, 0.0109547 * 0.75, 0.0109547 * 0.5, 0.0109547 * 0.25]
# rate2km = [8.761284524166374 * .2, 8.943103774908446 * .2, 9.309721342216584* .2, 9.548282786317728 * .2]
# rate2km37 = [8.674916189409016 * .2, 8.92755864006063 * .2, 9.172632076512993 * .2, 9.410495884848318 * .2]
# rate2km711 = [8.626502717320037* .2, 8.873978802774404* .2, 9.090271994619997* .2, 9.278196445692389* .2,]
# # plt.plot(p, rate2km37, marker="o", label="QPC (3,7)")
# plt.plot(p, rate2km, marker="s", label="Adaptive")
# plt.plot(p, rate2km711, marker="^", label="QPC (7,11)")
# plt.plot(p, rate2km37, marker="o", label="QPC (3,7)")
# plt.xlabel("Expected Error Probability $\mathbb{E}[e]$")
# plt.ylabel("Key Rate (Kbps)")
# plt.legend()
# plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# plt.grid(color='0.90', linestyle='--', linewidth=0.2)
# fig = plt.gcf()
# fig.savefig("rate2kmdepo.pdf")
# plt.show()

# # depo coeff 1
# plt.figure(figsize=(5, 4))
# pp = [0.131356, 0.131356*0.75, 0.131356*0.5, 0.131356*0.25]
# rate2km = [8.761284524166374* .2, 9.109620805586799 * .2, 9.130356883742024 * .2, 9.170526378240924 * .2]
# rate2km37 = [8.674916189409016* .2, 8.83770451271017* .2, 8.974099349738939* .2, 9.082347891226284* .2]
# rate2km711=[8.626502717320037* .2, 8.896896969619954* .2, 9.156990074291848* .2, 9.36838380068099* .2]
# plt.plot(pp, rate2km, marker="s", label="Adaptive")
# plt.plot(pp, rate2km711, marker="^", label="QPC (7,11)")
# plt.plot(pp, rate2km37, marker="o", label="QPC (3,7)")
# plt.xlabel("Expected Loss Probability $\mathbb{E}[1-\eta]$")
# plt.ylabel("Key Rate (Kbps)")
# plt.legend()
# plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# plt.grid(color='0.90', linestyle='--', linewidth=0.2)
# fig = plt.gcf()
# fig.savefig("rate2kmloss.pdf")
# plt.show()

# # loss coeff 1
# plt.figure(figsize=(5, 4))
# p = [0.0109547, 0.0109547 * 0.75, 0.0109547 * 0.5, 0.0109547 * 0.25]
# rate3km = [8.304944359606905 * .2, 8.48220133889273 * .2, 8.711776078357612 * .2, 9.031794393188578 * .2]
# rate3km37 = [8.177466081546392 * .2, 8.480966273174666 * .2, 8.783872652262739 * .2, 9.088000922058898 * .2]
# rate3km711 = [7.843724967493999* .2, 8.199519013147318* .2, 8.53374172105902* .2, 8.843967848056804* .2,]
# plt.plot(p, rate3km, marker="s", label="Adaptive")
# plt.plot(p, rate3km711, marker="^", label="QPC (7,11)")
# plt.plot(p, rate3km37, marker="^", label="QPC (3,7)")
# plt.xlabel("Expected Error Probability $\mathbb{E}[e]$")
# plt.ylabel("Key Rate (Kbps)")
# plt.legend()
# plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# plt.grid(color='0.90', linestyle='--', linewidth=0.2)
# fig = plt.gcf()
# fig.savefig("rate3kmdepo.pdf")
# plt.show()

# # depo coeff 1
# plt.figure(figsize=(5, 4))
# pp = [0.17372, 0.17372*0.75, 0.17372*0.5, 0.17372*0.25]
# rate3km = [8.304944359606905 * .2, 8.334331145375367* .2, 8.469109821550731* .2, 8.635949798065566* .2]
# rate3km37 = [8.177466081546392 * .2, 8.348303880712622 * .2, 8.49977196441736* .2, 8.626906838457442* .2]
# rate3km711 = [7.843724967493999* .2,8.084533478742516* .2, 8.34272100950268* .2, 8.587305363787838* .2]
# plt.plot(pp, rate3km, marker="s", label="Adaptive")
# plt.plot(pp, rate3km711, marker="^", label="QPC (7,11)")
# plt.plot(pp, rate3km37, marker="o", label="QPC (3,7)")
# plt.xlabel("Expected Loss Probability $\mathbb{E}[1-\eta]$")
# plt.ylabel("Key Rate (Kbps)")
# plt.legend()
# plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}'))
# plt.grid(color='0.90', linestyle='--', linewidth=0.2)
# fig = plt.gcf()
# fig.savefig("rate3kmloss.pdf")
# plt.show()