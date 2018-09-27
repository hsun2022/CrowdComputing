function result = MajorityVote(model)
%MAJORTYVOTE Summary of this function goes here
%   Detailed explanation goes here
Ntask = model.Ntask; 
Nwork = model.Nwork;
Ndom = model.Ndom;
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
for task_j=1:Ntask
    Lj=LabelTask{task_j};
    dom_j=unique(Lj);
    Ndom_j=length(dom_j);
    for k=1:Ndom_j
        count(k,1)=dom_j(k);
        count(k,2)=sum(Lj==dom_j(k));
    end
    count=sortrows(count,-2);
    soft_labels(1:Ndom_j,task_j)=count(1:Ndom_j,1);
    soft_labels_count(1:Ndom_j,task_j)=count(1:Ndom_j,2);
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
result.ans_labels=ans_labels;
result.no_ans_labels=no_ans_labels;
result.ans_multiRangeIndex=ans_multiRangeIndex;
result.soft_labels=soft_labels;
result.soft_labels_count=soft_labels_count;
result.accuracy=[];
if ~isempty(model.true_labels)
    result.accuracy=sum(~(ans_labels-model.true_labels))/Ntask;
end



end

    