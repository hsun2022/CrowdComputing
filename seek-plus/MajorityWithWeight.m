function result = MajorityWithWeight(model)
%MAJORTYVOTE Summary of this function goes here
%   Detailed explanation goes here
maxIter = 100;
TOL =1e-6;
Ntask = model.Ntask; 
Nwork = model.Nwork;
Ndom = model.Ndom;
L = model.L;
NeibTask = model.NeibTask; 
NeibWork = model.NeibWork;
LabelTask = model.LabelTask;
LabelWork = model.LabelWork;
LabelDomain =model.LabelDomain;
ans_labels=zeros(1,Ntask);
soft_labels=zeros(1,Ntask);
soft_labels_count=zeros(1,Ntask);
no_ans_labels=[];
ans_multiRangeIndex=[];
err = Inf;
Ability = ones(1,Nwork);
for iter = 1:maxIter
    if err<TOL
        break;
    end
    
    for task_j=1:Ntask
        Lj=LabelTask{task_j};
        dom_j=unique(Lj);
        Ndom_j=length(dom_j);
        Workj = NeibTask{task_j};
        Nworkj = length(Workj);
        count = [dom_j',zeros(Ndom_j,1)];
        for i = 1:Nworkj
            count(dom_j==Lj(i),2)= count(dom_j==Lj(i),2)+Ability(Workj(i))*1;
        end
        count=sortrows(count,-2);
        soft_labels(1:Ndom_j,task_j)=count(1:Ndom_j,1);
        soft_labels_count(1:Ndom_j,task_j)=count(1:Ndom_j,2);
        messCount{task_j}.soft_labels=soft_labels;
        messCount{task_j}.soft_labels_count=soft_labels_count;
        max=0;
        maxIndex=[];
        for i=1:size(count,1)
            if count(i,2)>max
                maxIndex=count(i,1);
                max=count(i,2);
            elseif count(i,2)==max
                maxIndex=[maxIndex count(i,1)];
            end
        end
        ans_num=length(maxIndex);
        if ans_num>1
            ans_labels(task_j)=maxIndex(1);%count一致，选第一个作为答案
            %ans_labels(task_j)=maxIndex(unidrnd(ans_num));%随机选了一个答案
            ans_multiRangeIndex=[ans_multiRangeIndex task_j];
        elseif ans_num==1
            ans_labels(task_j)=maxIndex(1);
        else
            ans_labels(task_j)=LabelDomain(unidrnd(Ndom));
            no_ans_labels=[no_ans_labels tasks_j];%没有人做task_j；
        end
    end
    Ability_new = ones(1,Nwork);
    for work_i = 1:Nwork
        Li = LabelWork{work_i};
        Taski = NeibWork{work_i};
        tem = 0;
        for j  = 1:length(Taski);
            task_index=find(soft_labels(:,Taski(j))==Li(j),1);
            tem=tem+soft_labels_count(task_index,Taski(j))/sum(soft_labels_count(:,Taski(j)));
        end
        Ability_new(work_i) = tem/length(Taski);    
    end
    err = norm(Ability_new-Ability);
    Ability  = Ability_new;
end
Simplicity=zeros(1,Ntask);
for task_j=1:Ntask
    count=0;
    total=0;
    for i=1:length(NeibTask{task_j});
        work_i=NeibTask{task_j}(i);
        total=total+1;
        if L(task_j,work_i)==ans_labels(task_j)
            count=count+1;
        end
    end
    Simplicity(task_j)=count/total;
end

result.ans_labels=ans_labels;
result.Simplicity=Simplicity;
result.no_ans_labels=no_ans_labels;
result.ans_multiRangeIndex=ans_multiRangeIndex;
result.soft_labels=soft_labels;
result.soft_labels_count=soft_labels_count;
result.Ability = Ability;
result.accuracy=[];
if ~isempty(model.true_labels)
    result.accuracy=sum(~(ans_labels-model.true_labels))/Ntask;
end



end   

