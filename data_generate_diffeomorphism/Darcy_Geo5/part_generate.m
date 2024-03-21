num = 30;
part_num = [1:num]';
part_size = eye(0);

midpoint_x = randi([30,70],num,1);
midpoint_x = midpoint_x./10;

leftpoint_x = randi([0,20],num,1);
leftpoint_x = leftpoint_x./10;

rightpoint_x = randi([80,100],num,1);
rightpoint_x = rightpoint_x./10;

leftpoint_y = randi([40,60],num,1);
leftpoint_y = leftpoint_y./10;

rightpoint_y = randi([40,60],num,1);
rightpoint_y = rightpoint_y./10;

down_leftpoint_x = randi([20,40],num,1);
down_leftpoint_x = down_leftpoint_x./10;

down_rightpoint_x = randi([60,80],num,1);
down_rightpoint_x = down_rightpoint_x./10;


midpoint = [midpoint_x];
leftpoint = [leftpoint_x, leftpoint_y];
rightpoint = [rightpoint_x, rightpoint_y];
down_leftpoint = [down_leftpoint_x];
down_rightpoint = [down_rightpoint_x];

part_size = [midpoint,leftpoint,rightpoint,down_leftpoint,down_rightpoint];
%%%%%%% random scale
random_numbers = rand(num, 1);
random_numbers = 0.5 + 0.5 * random_numbers;
scaled = round(random_numbers, 1); %*0+1
 
%%%%%%%%%%%%%

%%%%%%%%%%%%%
% min_scale = 3.5;
% max_scale = 8;
% step = 0.5;
% sample_size = 15;
% 
% num_scales = (max_scale - min_scale) / step + 1;
% 
% scaled = zeros(sample_size * round(num_scales), 1);
% current_index = 1;
% for scale = min_scale:step:max_scale
%     % ?????????
%     scaled(current_index:current_index+sample_size-1) = round(scale, 1);
%     current_index = current_index + sample_size;
% end
%%%%%%%%%%%%%

data = [part_num,part_size.*scaled,scaled];
csvwrite('geo_data.csv',data);
% results = solve_pde(part_size,cof);