function model = crowd_model(L,varargin)

%%parameters....
Ntask =size(L,1);
Nwork=size(L,2);
%neiborhods
NeibTask = cell(1,Ntask);
for task_j = 1:Ntask
    NeibTask{task_j} = find(L(task_j,:)); 
end
NeibWork = cell(Nwork,1);
for work_i = 1:Nwork
    NeibWork{work_i} = find(L(:,work_i))';
end

model=init_model();

LabelTask=cell(1,Ntask);
for task_j = 1:Ntask, 
    LabelTask{task_j} = L(task_j, NeibTask{task_j}); % the labels of the same task should have the same dimension
end

LjDomain=cell(1,Ntask);

for task_j=1:Ntask
    LjDomain{task_j} = unique(LabelTask{task_j});
end

LabelWork=cell(Nwork,1);
for work_i = 1:Nwork, 
    LabelWork{work_i} = L(NeibWork{work_i},work_i)'; % the labels of the same task should have the same dimension
end

LabelDomain = unique(L(L~=0));
model.LabelDomain = LabelDomain(:)';
model.Ndom = length(LabelDomain); 

model.LabelTask=LabelTask;
model.LabelWork=LabelWork;
model.LjDomain=LjDomain;
model.Ntask = Ntask;
model.Nwork = Nwork;
model.NeibTask = NeibTask;
model.NeibWork = NeibWork;
model.L=L;

if ~isempty(varargin)
    model.true_labels=varargin{1};
end

end

function model = init_model()

              model.L= [];       %: [5000x5000 double]
      model.LabelTask= [];       %: {1x5000 cell}
       model.LjDomain= [];
      model.LabelWork= [];       %: {1x5000 cell}
    model.LabelDomain= [];       %: [1 2]
          model.Ntask= [];       %: 5000
          model.Nwork= [];       %: 5000
           model.Ndom= [];       %: [1 2]                  
       model.NeibTask= [];       %: {1x5000 cell}
       model.NeibWork= [];       %: {1x5000 cell}
    model.true_labels= [];       %: [1x5000 double]

        return;
end

