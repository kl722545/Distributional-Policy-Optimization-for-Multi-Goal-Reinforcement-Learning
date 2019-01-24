clear;
clc;
% load from data.
color_order = get(groot,'defaultAxesColorOrder');
default_orange = color_order(2,:);
default_blue = color_order(1,:);


i = 0;
filename = ['DACER' num2str(i)]
path = 'Reacher-v2\DACER\';
% filename = ['DACER_Final']
load([path filename '.mat'])
v = VideoWriter([path filename '.mp4'],'MPEG-4');
v.Quality = 100;
v.FrameRate = 50;

action_low = -2.;
action_high = 2.;
x = action_low:0.001:action_high;
atoms = linspace(double(categorical_v_min),double(categorical_v_max),categorical_num_atom);
observations = permute(observations, [2 3 4 1]);
policy_images = zeros(size(observations),'like',observations);
value_images = zeros(size(observations),'like',observations);

for i = 1:length(rewards)
    % Get some info.
    m = means(i,:);
    s = stddevs(i,:);
    action = actions(i,:);
    figure(1);
    plot3(action(1), action(2), 0, 'r*');
    hold on;
    t = linspace(0, 2 * pi);
    e = bsxfun(@plus, s' .* [cos(t) ;sin(t)], m');
    plot3(e(1,:), e(2,:),zeros(1, length(t)),'color',default_orange);
    for side = 1:2
        y = normpdf(x,m(side),s(side));   
        if side == 1
            plot3(x, ones(1, length(x)) * action_high, y,'color',default_blue);
            plot3(action(side), action_high, normpdf(action(side),m(side),s(side)), 'r*');
        else
            plot3(ones(1, length(x)) * action_high, x, y,'color',default_blue);
            plot3(action_high, action(side), normpdf(action(side),m(side),s(side)), 'r*');
        end
    end

    axis([action_low action_high action_low action_high 0 3]);
    grid on;    
    hold off;
    position = get(gcf,'Position');
    position(1) = 200;
    position(2) = 200;
    position(3) = 500;
    position(4) = 500;
    set(gcf,'Position',position);
    
    title('policy distribution','Interpreter','latex','fontsize',12)
    xlabel('$a_1$','Interpreter','latex');
    ylabel('$a_2$','Interpreter','latex');
    zlabel('probability density $\boldmath{\pi}(\mathbf{a})$','Interpreter','latex','fontsize',12)
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