function result=SEEK_lnr_norm( model,KnowledgeMatrix)
%% parameters setting
maxIter=400;
TOL=1e-6;
global L NeibTask NeibWork LabelDomain Relation Ntask Nwork Ndom LabelTask LabelWork LjDomain prior_A prior_S variance_A variance_S
Ntask = model.Ntask; 
Nwork = model.Nwork;
Ndom = model.Ndom;
NeibTask = model.NeibTask; 
NeibWork = model.NeibWork;
LabelDomain =model.LabelDomain;
Relation=KnowledgeMatrix;
L=model.L; 
LabelTask=model.LabelTask;
LabelWork=model.LabelWork;
LjDomain=model.LjDomain;
majority=MajorityVote(model);
ans_labels=majority.ans_labels;
result.majority_anslabel=ans_labels;
if ~isempty(model.true_labels)
    result.majority_accuracy=sum(~(ans_labels-model.true_labels))/Ntask;
end

prior_A=zeros(1,Nwork);
prior_S=zeros(1,Ntask);
variance_A=1;
variance_S=1;

%% main iteration
err = NaN;%³õÊ¼»¯Îó²î
Ability = zeros(1, Nwork);
Simplicity = zeros(1, Ntask);
% goal = -Inf;
% for i = 1:10000
%     A = rand(1,Nwork);
%     S = rand(1, Ntask);
%     goal_new = likehood(A, S);
%     if goal_new > goal
%         goal = goal_new;
%         Ability = A;
%         Simplicity = S;
%     end
% end

% Ability = prior_A;
% Simplicity = prior_S;
% for work_i=1:Nwork 
%     Ability(work_i)=sum(~(model.L(NeibWork{work_i},work_i)-ans_labels(NeibWork{work_i})'))/length(NeibWork{work_i});    
% end
% Ability=(Ability-mean(Ability))*10;
main_Q=-Inf;
for iter = 1:maxIter  
    if err<TOL;%Îó²îÂú×ãÊÕÁ²ÒªÇó
        break;
    elseif iter>=maxIter
        info='iter in mainFunction reached to maxIter'
        break;
    end
    p_lt=Estep(Ability,Simplicity);
    Q_int=Qfun(Ability,Simplicity,p_lt);
    [Ability_tem,Simplicity_tem]=Mstep(Ability,Simplicity,p_lt);
    Ability=Ability_tem;
    Simplicity=Simplicity_tem;
    Q_inc=Qfun(Ability,Simplicity,p_lt);
    if Q_inc<Q_int
        info='error in Mstep'
        return;
    end
    Q_exp=Qfun(Ability,Simplicity,Estep(Ability,Simplicity));
    Q_inc=Qfun(Ability,Simplicity,p_lt);
    goal=likehood(Ability,Simplicity);
    err_EstepToGoal=goal-Q_exp
%     if Q_exp<Q_inc
%         info='error in Estep'
%         Q_exp
% %         p_lt=p_lt';
% %         p_lt2=p_lt2';
% %         save('p_lt.mat','p_lt')
% %         save('p_lt2.mat','p_lt2')
%         return;
%     end
    main_newQ=Q_exp;
    err=abs((main_newQ-main_Q)/main_Q)
    
    main_Q=main_newQ
end
L_anslabel=ones(1,Ntask);
p_lt=Estep(Ability,Simplicity);
for task_j=1:Ntask
    L_anslabel(task_j)=LjDomain{task_j}(find(p_lt{task_j}==max(p_lt{task_j}),1));     
end
if ~isempty(model.true_labels)
    result.accuracy=sum(~(L_anslabel-model.true_labels))/Ntask;
    result.FaultLabelIndex=find(L_anslabel-model.true_labels);
end
result.anslabel=L_anslabel;
result.Simplicity=Simplicity;
result.Ability=Ability;
result.softlabel = LjDomain;
result.softlabel_count=p_lt;
end

function p=p_Lij_LtAiSj(work_i,task_j,kth_label,ai,sj)
global L Relation  LjDomain 
p=0;
uniform=0;
LjD=LjDomain{task_j};
NumLjD = length(LjD);
for k=1:NumLjD
    if LjD(k)==kth_label
        uniform=uniform+(NumLjD-1)*exp(ai+sj+Relation(LjD(k),kth_label))+0.01;
    else
        uniform=uniform+exp(Relation(LjD(k),kth_label));
    end
end
lij=L(task_j,work_i);
if lij==0
    errlocal_p_Lij_LtAiSj=[task_j,work_i]
    return;
end

if lij==kth_label
    p=((NumLjD-1)*exp(ai+sj+Relation(lij,kth_label))+0.01)/uniform;
else
    p=exp(Relation(lij,kth_label))/uniform;
end
end

function goal=likehood(A,S)
global Ntask LjDomain NeibTask Nwork prior_A prior_S variance_A variance_S
goal=0;
for task_j=1:Ntask
    LjD=LjDomain{task_j};
    NumLjD=length(LjD);
    sum=0;
    for k=1:NumLjD
        tem=1;
        for i=1:length(NeibTask{task_j})
            work_i=NeibTask{task_j}(i);
            tem=tem*p_Lij_LtAiSj(work_i,task_j,LjD(k),A(work_i),S(task_j));
        end
        sum=sum+tem;
    end
    goal=goal+log(sum);
end
sum_LjD=0;
for task_j=1:Ntask
    LjD=LjDomain{task_j};
    NumLjD=length(LjD);
    sum_LjD=sum_LjD-log(NumLjD);
end
sum_i=0;
for work_i=1:Nwork
    sum_i=sum_i+log(normpdf(A(work_i),prior_A(work_i),variance_A));
end
sum_j=0;
for task_j=1:Ntask
    sum_j=sum_j+log(normpdf(S(task_j),prior_S(task_j),variance_S));
end
goal=goal+sum_i+sum_j+sum_LjD;
end

function p_lt=Estep(A,S)
global LjDomain Ntask NeibTask
p_lt=cell(1,Ntask);
for task_j=1:Ntask
    LjD=LjDomain{task_j};
    NumLjD=length(LjD);
    p_lt{task_j}=zeros(1,NumLjD);
    for k=1:NumLjD
        tem=1;
        for i=1:length(NeibTask{task_j})
            work_i=NeibTask{task_j}(i);
            %lij=L(task_j,work_i);
            tem=tem*p_Lij_LtAiSj(work_i,task_j,LjD(k),A(work_i),S(task_j));
        end
        p_lt{task_j}(k)=tem;
    end
    p_lt{task_j}=p_lt{task_j}/sum(p_lt{task_j});
end
end

function [g_A,g_S]=update_diff(A,S,p_lt)
global Nwork Ntask LjDomain NeibWork Relation L prior_A prior_S variance_A variance_S
g_A=zeros(1,Nwork);
g_S=zeros(1,Ntask);
dL=zeros(Ntask,Nwork);
for work_i=1:Nwork
    for j=1:length(NeibWork{work_i})
        task_j=NeibWork{work_i}(j);
        LjD=LjDomain{task_j};
        lij=L(task_j,work_i);
        if lij==0
            errCall_lij=[task_i,work_i] 
            return;
        end
        kthLabel=find(LjD==lij,1);
        tem=0;
        NumLjD = length(LjD);
        for k=1:NumLjD
            sum_ii=(NumLjD-1)*exp(A(work_i)+S(task_j)+Relation(LjD(k),LjD(k)));
            sum_ij=0;
            for k2=1:NumLjD
                if k2~=k
                    sum_ij=sum_ij+exp(Relation(LjD(k2),LjD(k)));
                end
            end
            if k==kthLabel
                tem=tem+p_lt{task_j}(k)*(sum_ii/(sum_ii+0.01)-sum_ii/(sum_ij+sum_ii+0.01));
            else
                tem=tem-p_lt{task_j}(k)*sum_ii/(sum_ij+sum_ii+0.01);
            end
        end
        dL(task_j,work_i)=tem;
    end
end
% dL=dL
g_A=sum(dL,1)+(prior_A-A)/variance_A^2;
g_S=sum(dL,2)'+(prior_S-S)/variance_S^2;
end


function Q=Qfun(A,S,p_lt)
%¸¨Öúº¯Êý
global NeibWork  Nwork  Ntask LjDomain prior_A prior_S variance_A variance_S
Q=0;
sum_main=0;
for work_i=1:Nwork
    for j=1:length(NeibWork{work_i})
        task_j=NeibWork{work_i}(j);
        LjD=LjDomain{task_j};
        NumLjD=length(LjD);
        %% µ÷ÊÔ
        if length(NeibWork{work_i})<1
            disp(work_i);
            error('err in length of NeibWork{work_i}');
        end
        %%
        for k=1:NumLjD
            sum_main=sum_main+p_lt{task_j}(k)*log(p_Lij_LtAiSj(work_i,task_j,LjD(k),A(work_i),S(task_j)));
        end
    end
end
sum_LjD=0;
sum_Q=0;
for task_j=1:Ntask
    LjD=LjDomain{task_j};
    NumLjD=length(LjD);
    sum_LjD=sum_LjD-log(NumLjD);
    for k=1:NumLjD
        sum_Q=sum_Q-p_lt{task_j}(k)*log(p_lt{task_j}(k));
    end
end
sum_i=0;
for work_i=1:Nwork
    sum_i=sum_i+log(normpdf(A(work_i),prior_A(work_i),variance_A));
end
sum_j=0;
for task_j=1:Ntask
    sum_j=sum_j+log(normpdf(S(task_j),prior_S(task_j),variance_S));
end
%likehood=sum;
% sum_main
% sum_i
% sum_j
%sum_LjD
% sum_Q
Q=sum_main+sum_Q+sum_i+sum_j+sum_LjD;
end

function [Ability,Simplicity]=Mstep(oldAbility,oldSimplicity,p_lt)
global Nwork Ntask LjDomain NeibTask LabelTask Relation
err = NaN;
maxIter=400;
TOL=1e-6;
Ability=oldAbility;
Simplicity=oldSimplicity;
Q=Qfun(Ability,Simplicity,p_lt);
for iter =1:maxIter
    if err<TOL;%Îó²îÂú×ãÊÕÁ²ÒªÇó
        break;
    elseif iter>=maxIter
        iter
        info='iter in argQ reached to maxIter'
        break;
    else
        for task_j = 1:Ntask
            LjD = LjDomain{task_j};
            NumLjD = length(LjD);
            workerList = NeibTask{task_j};
            labelList = LabelTask{task_j};
            Nworker_j = length(workerList);
            for i = 1:Nworker_j
                worker_i = workerList(i);
                lij = labelList(i);
                B{task_j,worker_i} = ones(1,NumLjD);
                for k = 1:NumLjD
                    groundTruth = LjD(k);
                    tem = 0;
                    for k2 = 1:NumLjD
                        if k2 == k
                            tem = tem + (NumLjD-1)*exp(Ability(worker_i)+Simplicity(task_j)+Relation(groundTruth,groundTruth));
                        else
                            tem = tem + exp(Relation(LjD(k2),groundTruth));
                        end
                    end
                    B{task_j,worker_i}(k) = 1/tem;
                end
            end
        end
        newAbility = zeros(1,Nwork);
        newSimplicity = zeros(1,Ntask);
        for task_j = 1:Ntask
            LjD = LjDomain{task_j};
            NumLjD = length(LjD);
            workerList = NeibTask{task_j};
            labelList = LabelTask{task_j};
            Nworker_j = length(workerList);
            for i = 1:Nworker_j
                worker_i = workerList(i);
                lij = labelList(i);
                tem = 0;
                for k = 1:NumLjD
                    groundTruth = LjD(k);
                    tem2 = 0;
                    for k2 = 1:NumLjD
                        if LjD(k2) ~= groundTruth
                            tem2 = tem2 + exp(Relation(LjD(k2),groundTruth));
                        end
                    end
                    if groundTruth == lij
                        tem = tem + p_lt{task_j}(k);
                    else
                        tem = tem - p_lt{task_j}(k)*(1-B{task_j,worker_i}(k)*tem2);
                    end
                end
                newAbility(worker_i) = newAbility(worker_i) + tem;
                newSimplicity(task_j) = newSimplicity(task_j) + tem;
            end
        end
        err = sum(abs(Ability-newAbility)) + sum(abs(Simplicity-newSimplicity));
        Ability = (Ability+newAbility)/2;
        Simplicity = (Simplicity+newSimplicity)/2;
    end   
end
end

% function aroundQ(oldA,oldS,p_lt)
% Q_mid=Qfun(oldA,oldS,p_lt);
% [g_A,g_S]=update_diff(oldA,oldS,p_lt);
% alpha=1;
% for iter =1:10
%     A=oldA+alpha*g_A;
%     S=oldS+alpha*g_S;
%     Q_tem=Qfun(A,S,p_lt)
%     alpha=alpha/2;
% end
% Q_mid
% for iter =1:10
%     A=oldA-alpha*g_A;
%     S=oldS-alpha*g_S;
%     Q_tem=Qfun(A,S,p_lt)
%     alpha=alpha*2;
% end
% end