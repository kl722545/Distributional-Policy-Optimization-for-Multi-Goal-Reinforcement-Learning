clear;
clc;

path = 'InvertedPendulum-v2\DPPO\';
i = 0;
filename = ['DPPO' num2str(i)]
% filename = ['DPPO_Final']
load([path filename '.mat']);
savepath = sprintf('.\\InvertedPendulum-v2\\DPPO\\%s',filename);


atoms = linspace(double(categorical_v_min),double(categorical_v_max),categorical_num_atom);
action_low = -15.;
action_high = 15.;
x = action_low:0.001:action_high;
observations = permute(observations, [2 3 4 1]);

for i = 1:length(rewards)
    datafilename = sprintf('%s\\snapshot\\%d.png',savepath,i);
    imwrite(observations(:,:,:,i),datafilename);
    mean = means(i);
    stddev = stddevs(i);
    action = actions(i);
    figure(1);
    plot(x, normpdf(x,mean,stddev));
    hold on;
    plot(action , normpdf(action,mean,stddev),'r*');
    axis([action_low action_high 0 0.5]);
    hold off;
    position = get(gcf,'Position');
    position(1) = 200;
    position(2) = 200;
    position(3) = 500;
    position(4) = 500;
    set(gcf,'Position',position);
    title('policy distribution','Interpreter','latex','fontsize',12)
    xlabel('action $a$','Interpreter','latex','fontsize',12)
    ylabel('probability density $\boldmath{\pi}(a)$','Interpreter','latex','fontsize',12)
    datafilename = sprintf('%s\\policy\\%d.png',savepath,i);
    imwrite(frame2im(getframe(gcf)),datafilename);

    figure(2);
    Y = value_dists(i,:);
    stairs(atoms,Y);
    axis([double(categorical_v_min) double(categorical_v_max) 0. 0.5]);
    position = get(gcf,'Position');
    position(1) = 1000;
    position(2) = 200;
    position(3) = 500;
    position(4) = 500;
    set(gcf,'Position',position);
    
    title('value distribution','Interpreter','latex','fontsize',12);
    xlabel('value $v$','Interpreter','latex','fontsize',12);
    ylabel('probability mass $p_v$','Interpreter','latex','fontsize',12);
    datafilename = sprintf('%s\\value\\%d.png',savepath,i);
    imwrite(frame2im(getframe(gcf)),datafilename);
end


