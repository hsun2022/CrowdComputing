clear;
clc;

Ntask = 100;
Nworker = 10;
Ndom = 7;
Redun = 5;
ndom = 3;
p0 = 0.05;
p1 = 0.75;
times = 30;
x = 10:5:100;
num = length(x);
MVresultSet = zeros(10,num);
DSresultSet = zeros(10,num);
SEEKresultSet = zeros(10,num);
noKnowledgeSEEKresultSet = zeros(10,num);
for j = 1:times
    for i = 1:num
        [MVresult,DSresult,SEEKresult,noKnowledgeSEEKresult] = test_simulation(Ntask,x(i),Ndom,Redun,ndom,p0,p1);
        MVresultSet(j,i) = MVresult;
        DSresultSet(j,i) = DSresult;
        SEEKresultSet(j,i) = SEEKresult;
        noKnowledgeSEEKresultSet(j,i) = noKnowledgeSEEKresult;
    end
end

MVresultMean = mean(MVresultSet,1);
DSresultMean = mean(DSresultSet,1);
SEEKresultMean = mean(SEEKresultSet,1);
noKnowledgeSEEKresultMean = mean(noKnowledgeSEEKresultSet,1);
plot(x,MVresultMean,'-b*');
hold on;
plot(x,DSresultMean,'-rx');
plot(x,SEEKresultMean,'-gs');
plot(x,noKnowledgeSEEKresultMean,'-ko')
legend('MV','DS','SEEK','SEEKsimple');
%title('test for Ntask');
xlabel('the count of worker');
