clear;
clc;
DPPO = load(['InvertedPendulum-v2\DPPO\returns.mat']);
PPO = load(['InvertedPendulum-v2\PPO\returns.mat']);
DACER = load(['Reacher-v2\DACER\returns.mat']);
ACER = load(['Reacher-v2\ACER\returns.mat']);
color_order = get(groot,'defaultAxesColorOrder');
default_orange = color_order(2,:);
light_orange = [1.0000, 0.6275, 0.4784];
default_blue = color_order(1,:);
light_blue = [0.5843, 0.8157, 0.9882];

figure;
ppo_line = plot_return(PPO,default_orange,light_orange,500);
hold on;
dppo_line = plot_return(DPPO,default_blue,light_blue,500);
title('InvertedPendulum-v2');
legend([dppo_line ppo_line],'Distri. PPO','PPO','Location','southeast')
axis([0,inf,0,1200]);

figure;
ppo_line = plot_return(ACER,default_orange,light_orange,500);
hold on;
dppo_line = plot_return(DACER,default_blue,light_blue,500);
title('Reacher-v2');
legend([dppo_line ppo_line],'Distri. ACER','ACER','Location','southeast')
%axis([0,inf,0,1200]);

