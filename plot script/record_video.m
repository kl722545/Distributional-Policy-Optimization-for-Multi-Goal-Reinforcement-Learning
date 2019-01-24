clear;
clc;
i = 9;
path = 'InvertedPendulum-v2\DPPO\';
filename = ['DPPO' num2str(i)]
%filename = ['DPPO_Final']
load([path filename '.mat'])
v = VideoWriter([path filename '.mp4'],'MPEG-4');
v.Quality = 100;
v.FrameRate = 50;
atoms = linspace(double(categorical_v_min),double(categorical_v_max),categorical_num_atom);
action_low = -15.;
action_high = 15.;
x = action_low:0.001:action_high;
observations = permute(observations, [2 3 4 1]);
policy_images = zeros(size(observations),'like',observations);
value_images = zeros(size(observations),'like',observations);
for i = 1:length(rewards)
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
    policy_images(:,:,:,i) = frame2im(getframe(gcf));

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
    value_images(:,:,:,i) = frame2im(getframe(gcf));
end

frames = cat(2, observations, policy_images, value_images);
open(v);
writeVideo(v,frames);
close(v);
