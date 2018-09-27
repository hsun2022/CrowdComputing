clear;
tic
load('L_wordnet.mat');
load('groundFitData.mat');
load('knowledgeMatrix.mat'); 
p1=0.05;p=0.75;
maxIter = 50;
[N,M] = size(knowledgeMatrix);
proMatrix = zeros(N,M);
for i = 1:N
    for j = 1:M
        if knowledgeMatrix(i,j)>0 && i~=j
            proMatrix(i,j) = log(p/(1-p));
            proMatrix(j,i) = log(p1/(1-p1));
        end
    end
end
model = crowd_model(L_wordnet, groundFitData);
Ntask = model.Ntask;
ratioList = 0:0.05:0.5;
accuracyMatrix = zeros(length(ratioList),maxIter);
for i = 1:length(ratioList)
    ratio = ratioList(i);
    groundTruthNumber = round(Ntask*ratio);
    for iter = 1:maxIter
        randList = randperm(Ntask);
        groundTruthFlag = zeros(1,Ntask);
        for j = 1:groundTruthNumber
            groundTruthFlag(randList(j)) = 1;
        end
        result = SEEK_lnr_norm_semi(model,proMatrix,groundTruthFlag);
        accuracyMatrix(i,iter) = result.accuracy_unlabeled;
        disp(['retio=',num2str(ratio),' accuracy_unlabeled=',num2str(accuracyMatrix(i,iter))]);
    end
end
accuracyMeanList = mean(accuracyMatrix,2);
accuracyStdList = std(accuracyMatrix,0,2);
accuracyMaxList = accuracyMeanList + accuracyStdList;
accuracyMinList = accuracyMeanList - accuracyStdList;

% plot(ratioList,accuracyMeanList,'-b*');
% hold on
% plot(ratioList,accuracyMaxList,'-r*');
% plot(ratioList,accuracyMinList,'-r*');
errorbar(ratioList,accuracyMeanList,accuracyStdList);
title('Performance of SEEK on Different Ground Truth Ratio');
xlabel('ground truth ratio');
ylabel('accuracy');


toc