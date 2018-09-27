clear;
clc;
Ntask = 100;
Nworker = 10;
Ndom = 7;
Redun = 5;
ndom = 3;


 [L,groundtruth] = L_simulation_noNoise(Ntask,Nworker,Ndom,Redun,ndom);
 model = crowd_model(L,groundtruth);
 Ntask = model.Ntask;

p0Set = 0.05/32:0.05/32:0.05;
p1Set = 0.5:0.025:0.95;
[x,y]= meshgrid(p0Set,p1Set);
[n,m] = size(x);
SEEKresultSet = zeros(n,m);
for x1 = 1:n
    for x2 = 1:m
        p0 = x(x1,x2);
        p1 = y(x1,x2);
        Matrix = ones(Ndom,Ndom)*log(p0/(1-p0));
        for i = 1:Ndom
            for j = 1:Ndom
                if i == j
                    Matrix(i,j) = 0;
                elseif i == floor(j/2)
                    Matrix(i,j) = log(p1/(1-p1));
                    Matrix(j,i) = log(p0/(1-p0));
                end
            end
        end
        result = SEEK_lnr_norm_semi(model,Matrix,zeros(1,Ntask));
        SEEKresultSet(x1,x2) =result.accuracy_unlabeled ;
        disp(result.accuracy_unlabeled);
    end        
end
subplot(2,1,1);

surf(x,y,SEEKresultSet);

p0Set = 0.15/32:0.15/32:0.15;
p1Set = 0.5:0.025:0.95;
[x,y]= meshgrid(p0Set,p1Set);
[n,m] = size(x);
SEEKresultSet = zeros(n,m);
for x1 = 1:n
    for x2 = 1:m
        p0 = x(x1,x2);
        p1 = y(x1,x2);
        Matrix = ones(Ndom,Ndom)*log(p0/(1-p0));
        for i = 1:Ndom
            for j = 1:Ndom
                if i == j
                    Matrix(i,j) = 0;
                elseif i == floor(j/2)
                    Matrix(i,j) = log(p1/(1-p1));
                end
            end
        end
        result = SEEK_lnr_norm_semi(model,Matrix,zeros(1,Ntask));
        SEEKresultSet(x1,x2) =result.accuracy_unlabeled ;
        disp(result.accuracy_unlabeled);
    end        
end
subplot(2,1,2);
surf(x,y,SEEKresultSet);





