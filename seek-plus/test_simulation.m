function [MVresult,DSresult,SEEKresult,noKnowledgeSEEKresult]=test_simulation(Ntask,Nworker,Ndom,Redun,ndom,p0,p1)
Matrix = ones(Ndom,Ndom)*log(p0/(1-p0));
noKnowledgeMatrix = Matrix;
for i = 1:Ndom
    for j = 1:Ndom
        if i == j
            Matrix(i,j) = 0;
            noKnowledgeMatrix(i,j) = 0;
        elseif i == floor(j/2)
            Matrix(i,j) = log(p1/(1-p1));
        end
    end
end

addpath(genpath(pwd));

[L,groundtruth]=L_simulation_noNoise(Ntask,Nworker,Ndom,Redun,ndom);
model=crowd_model(L,groundtruth);
MV=MajorityVote(model);
SEEK=SEEK_lnr_norm_semi(model,Matrix,zeros(1,model.Ntask));
noKnowledgeSEEK = SEEK_lnr_norm_semi(model,noKnowledgeMatrix,zeros(1,model.Ntask));

model_zhou = crowd_model_zhou(L, 'true_labels',groundtruth);
DS = DawidSkene_crowd_model(model_zhou);

MVresult = MV.accuracy;
SEEKresult = SEEK.accuracy_unlabeled;
noKnowledgeSEEKresult = noKnowledgeSEEK.accuracy_unlabeled;
DSresult = 1-DS.error_rate;
end