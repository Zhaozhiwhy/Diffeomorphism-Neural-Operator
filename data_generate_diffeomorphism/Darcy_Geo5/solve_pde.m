function [results,c] = solve_pde(partsize,cof,scaled,scale_c)

model = createpde;
% P1 = [2,5,0,10,partsize(4),partsize(1),partsize(2),...
%           0,0,partsize(5),10,partsize(3)]';
P1 = [2,5,partsize(6),partsize(7),partsize(4),partsize(1),partsize(2),...
          0,0,partsize(5),10*scaled,partsize(3)]';
gm = [P1];
sf = 'P1';
ns = char('P1');
ns = ns';
g = decsg(gm,sf,ns);
geometryFromEdges(model,g);
% pdegplot(model,'EdgeLabels','on')
% ylim([-1.1,1.1])
axis equal
c = @(region,state)(cof(1)*sin(pi*(region.x/scaled/10))-cof(2).*(region.x/scaled).*(region.x/scaled-10)+2)*scale_c;

applyBoundaryCondition(model,'dirichlet','Edge',1:model.Geometry.NumEdges,'u',0);
specifyCoefficients(model,'m',0,...
                          'd',0,...
                          'c',c,...
                          'a',0,...
                          'f',1);
generateMesh(model,'Hmax',0.25*scaled);
results = solvepde(model);
%pdeplot(model,'XYData',c)

end

