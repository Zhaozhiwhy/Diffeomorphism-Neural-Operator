clear;
part_size_file = csvread('geo_new.csv');
part_num = part_size_file(:,1);
scaled = part_size_file(:,9);
part_size = part_size_file(:,2:8);
num = size(part_num);
num = num(1);
U = eye(0);
C = eye(0);

all_cof = randi([20,80],num,2);
all_cof = all_cof./100;

X = csvread('./data/x_data.csv');
Y = csvread('./data/y_data.csv');

%when resolution is larger than 512

% mesh_s= 1024;  
% X = reshape(X,mesh_s*mesh_s,num);
% Y = reshape(Y,mesh_s*mesh_s,num);
% X = X.';
% Y = Y.';

for i = 1:num
%     i=150;
    size = part_size(i,:);
    cof = all_cof(i,:);
    x = X(i,:);
    y = Y(i,:);
    
    if rand>0.5
        scale_c = 0.01+0.02*randi(50);
    else
        scale_c = 1+0.05*randi(60);
    end
    [results,cc] = solve_pde(size,cof,scaled(i),scale_c);
    xy = results.Mesh.Nodes;
    ui = results.NodalSolution;
    u = griddata(xy(1,:),xy(2,:),ui,x,y,'cubic');
    U = [U;u];
    c = cof(1)*sin(pi*(x/scaled(i)/10))-cof(2).*(x/scaled(i)).*(x/scaled(i)-10)+2;

    C = [C;c*scale_c];

end
csvwrite('./data/U.csv',U);
csvwrite('./data/C.csv',C);