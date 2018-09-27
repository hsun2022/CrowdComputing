function result = SEEK_lnr_norm_semi(model, KnowledgeMatrix,groundTruthFlag)
maxIter = 20;
TOL = 1e-6;
global L NeibTask NeibWork LabelDomain Relation Ntask Nwork Ndom LabelTask LabelWork LjDomain groundTruth Flag bias
Ntask = model.Ntask; 
Nwork = model.Nwork;
Ndom = model.Ndom;
NeibTask = model.NeibTask; 
NeibWork = model.NeibWork;
LabelDomain = model.LabelDomain;
Relation = KnowledgeMatrix;
L = model.L; 
LabelTask = model.LabelTask;
LabelWork = model.LabelWork;
LjDomain = model.LjDomain;
groundTruth = model.true_labels;
Flag = groundTruthFlag;
bias = 0.0001;
Nflag = sum(Flag);
majority = MajorityVote(model);
ansLabel = majority.ans_labels;
result.majorityAnsLabel = ansLabel;

if ~isempty(model.true_labels)
    result.majorityAccuracy = sum(~(ansLabel-model.true_labels))/Ntask;
end



err = NaN;
Ability = zeros(1,Nwork);
Simplicity = zeros(1,Ntask);
for iter = 1:maxIter
    if err < TOL
        break;
    elseif iter >=maxIter
%         disp('iter in main function reached to maxIter');
        break;
    end
    p_lt = Estep(Ability, Simplicity);
%     obj = Aux(Ability,Simplicity,p_lt);
%     disp(['object in Estep is: ',num2str(obj)]);
    [Ability_tem,Simplicity_tem] = Mstep(Ability, Simplicity, p_lt);
    err = (sum(abs(Ability_tem-Ability))+sum(abs(Ability_tem-Ability)))/(Ntask+Nwork);
    Ability = Ability_tem;
    Simplicity = Simplicity_tem;
%     obj = Aux(Ability,Simplicity,p_lt);
%     disp(['object in Mstep is: ',num2str(obj)]);
end
L_anslabel = ones(1,Ntask);
for task_j = 1:Ntask
    [~, I] = max(p_lt{task_j});
    L_anslabel(task_j) = LjDomain{task_j}(I);
end
if ~isempty(model.true_labels)
    result.accuracy = sum(~(L_anslabel-model.true_labels))/Ntask;
    result.FaultLabelIndex = find(L_anslabel-model.true_labels);
    result.accuracy_unlabeled = (result.accuracy-Nflag/Ntask)/(1-Nflag/Ntask);
end
result.anslabel=L_anslabel;
end

function p_lt = Estep(Ability, Simplicity)
global  NeibTask  LabelTask LjDomain Ntask Nwork Relation groundTruth Flag bias
p_lij = cell(Ntask,Nwork);
p_lt = cell(1,Ntask);
for task_j = 1:Ntask
    LjD = LjDomain{task_j};
    NumLjD = length(LjD);
    p_lt{task_j} = ones(1,NumLjD);
    if Flag(task_j) == 1
        for k = 1:NumLjD
            if LjD(k) == groundTruth(task_j)
                p_lt{task_j}(k) = 1;
            else
                p_lt{task_j}(k) = 0;
            end
        end
        continue;
    end
    workerList = NeibTask{task_j};
    labelList = LabelTask{task_j};
    Nworker_j = length(workerList);
    for i = 1:Nworker_j
        worker_i = workerList(i);
        lij = labelList(i);
        lij_index = find(LjD==lij,1);
        p_lij{task_j,worker_i} = zeros(NumLjD,NumLjD);
        for k = 1:NumLjD 
            for k2 = 1:NumLjD
                if k == k2
                    p_lij{task_j,worker_i}(k2,k) = (NumLjD-1+bias)*exp(Ability(worker_i)+Simplicity(task_j)+Relation(LjD(k2),LjD(k)));
                else
                    p_lij{task_j,worker_i}(k2,k) = exp(Relation(LjD(k2),LjD(k)));
                end
            end
            p_lij{task_j,worker_i}(:,k) = p_lij{task_j,worker_i}(:,k)/sum(p_lij{task_j,worker_i}(:,k));
            p_lt{task_j}(k) = p_lt{task_j}(k)*p_lij{task_j,worker_i}(lij_index,k);
        end
    end
    p_lt{task_j} = p_lt{task_j}/sum(p_lt{task_j});
end
end

function [Ability, Simplicity] = Mstep(oldAbility, oldSimplicity, p_lt)
global Ntask Nwork
err = NaN;
maxIter = 10;
TOL = 1e-6;
Ability = oldAbility;
Simplicity = oldSimplicity;

for iter = 1:maxIter
    if err < TOL
        return;
    elseif iter >= maxIter
%         disp('iter reachs to the maxIter in Mstep')
    end
    obj_old = Aux(oldAbility,oldSimplicity,p_lt);
    [A_gradient,S_gradient] = gradient(oldAbility,oldSimplicity,p_lt);
    alpha = 0.5;
    for iter2 = 1:maxIter
        if iter2 >= maxIter
            A_gradient
            S_gradient
            error('error in gradient');
        end
        Ability = oldAbility +alpha*A_gradient;
        Simplicity = oldSimplicity + alpha*S_gradient;
        obj = Aux(Ability,Simplicity,p_lt);
        if obj >= obj_old
            break;
        else
            alpha = alpha/4;
        end
    end
    err = (sum(abs(Ability-oldAbility))+sum(abs(Simplicity-oldSimplicity)))/(Ntask+Nwork);
    oldAbility = Ability;
    oldSimplicity = Simplicity;
end
end

function [A_gradient,S_gradient] = gradient(Ability,Simplicity,p_lt)
global Ntask Nwork LjDomain Relation NeibTask LabelTask bias
A_gradient = zeros(1,Nwork);
S_gradient = zeros(1,Ntask); 
for task_j = 1:Ntask
    LjD = LjDomain{task_j};
    NumLjD = length(LjD);
    workerList = NeibTask{task_j};
    labelList = LabelTask{task_j};
    NumWorker = length(workerList);
    for i = 1:NumWorker
        worker_i = workerList(i);
        lij = labelList(i);
        item = 0;
        for k = 1:NumLjD
            groundTruth = LjD(k);
            tem = 0;
            for k2 = 1:NumLjD
                if LjD(k2) == groundTruth
                    main_tem = (NumLjD-1+bias)*exp(Ability(worker_i)+Simplicity(task_j)+Relation(groundTruth,groundTruth));
                else
                    tem = tem + exp(Relation(LjD(k2),groundTruth));
                end
            end
            pro_main = main_tem/(main_tem+tem);
            if groundTruth == lij
                item = item + p_lt{task_j}(k) - p_lt{task_j}(k)*pro_main;
            else
                item = item -p_lt{task_j}(k)*pro_main;
            end
        end
        A_gradient(worker_i) = A_gradient(worker_i) + item;
        S_gradient(task_j) = S_gradient(task_j) + item;
    end
end
A_gradient = A_gradient - Ability;
S_gradient = S_gradient - Simplicity;
end

function obj = Aux(Ability,Simplicity,p_lt)
global NeibTask LabelTask Relation Nwork Ntask LjDomain Flag bias
part1 = 0;
part2 = 0;
part3 = 0;
B_reciprocal = cell(Ntask,Nwork);
for task_j = 1:Ntask
    workerList = NeibTask{task_j};
    labelList = LabelTask{task_j};
    LjD = LjDomain{task_j};
    NumLjD = length(LjD);
    NumWorker  = length(workerList);
    for k = 1:NumLjD
        if Flag(task_j) == 1
            break;
        end
        part1 = part1 - p_lt{task_j}(k)*log(NumLjD*p_lt{task_j}(k));
    end
    for i = 1:NumWorker
        worker_i = workerList(i);
        lij = labelList(i);
        B_reciprocal{task_j,worker_i} = zeros(1,NumLjD);
        for k = 1:NumLjD
            groundTruth = LjD(k);
            for k2 = 1:NumLjD
                if k2 == k
                    B_reciprocal{task_j,worker_i}(k) = B_reciprocal{task_j,worker_i}(k) + (NumLjD-1+bias)*exp(Ability(worker_i)+Simplicity(task_j)+Relation(groundTruth,groundTruth));
                else
                    B_reciprocal{task_j,worker_i}(k) = B_reciprocal{task_j,worker_i}(k) + exp(Relation(LjD(k2),groundTruth));
                end
            end
            part2 = part2 + p_lt{task_j}(k)*(Relation(lij,groundTruth)-log(B_reciprocal{task_j,worker_i}(k)));
            if lij == groundTruth
                part2 = part2 + p_lt{task_j}(k)*(Ability(worker_i)+Simplicity(task_j)+log(NumLjD-1+bias));
            end
        end
    end
end
for task_j = 1:Ntask
    part3 = part3 - Simplicity(task_j)^2/2;
end
for worker_i = 1:Nwork
    part3 = part3 - Ability(worker_i)^2/2;
end
obj = part1 + part2 + part3;
end