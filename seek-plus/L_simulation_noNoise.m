function [L,groundtruth] = L_simulation_noNoise(Ntask,Nworker,Ndom,Redun,ndom)
p0 = 0.05;
p1 = 0.75;
%prop = 0.5;
groundtruth = zeros(1,Ntask);
L = zeros(Ntask,Nworker);

for task_j = 1:Ntask
    labelSet = randperm(Ndom,ndom);
    groundtruth(task_j) = randsrc(1,1,labelSet);
    prob = zeros(1,ndom);
    for k = 1:ndom
        dom_k = labelSet(k);
        if isHypernym(dom_k,groundtruth(task_j))
            prob(k) = p1/(1-p1);
        elseif dom_k == groundtruth(task_j)
            prob(k) = ndom-1;
        else
            prob(k) = p0/(1-p0);
        end 
    end
    workerSet = randperm(Nworker,Redun);
    for i = 1:Redun
        worker_i = workerSet(i);
        prob = prob/sum(prob);
        L(task_j,worker_i) = randsrc(1,1,[labelSet;prob]);
    end
end
end

function flag = isHypernym(i,j)
% To confirm i is or not j's hypernym.
    if i > floor(j/2)
        flag = 0;
    elseif i == floor(j/2)
        flag = 1;
    else
        flag = isHypernym(i,floor(j/2));
    end
end

