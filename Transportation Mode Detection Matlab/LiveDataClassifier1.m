clear all
load('LiveClassifier.mat');
m = mobiledev;
m.Logging = 1;
pause(2);
y = 1;
n = 25;
v = 1;
pred = 0;
actual = 1;


while 1  
tic
z = 1;
x = 0.1;

while toc < 5
a(v) = m.Acceleration(1,1);    
b(v) = m.Acceleration(1,2);
c(v) = m.Acceleration(1,3);
plot(a);
hold on
plot(b);
plot(c);
hold off
title("Mode Prediction = " + pred + "Actual Mode " + actual);
pause(0.1);
v = v+1;        
if toc>x    
AX(z,1) = m.Acceleration(1,1);
AY(z,1) = m.Acceleration(1,2);
AZ(z,1) = m.Acceleration(1,3);
AvX(z,1) = m.AngularVelocity(1,1);
AvY(z,1) = m.AngularVelocity(1,2);
AvZ(z,1) = m.AngularVelocity(1,3);
MX(z,1) = m.MagneticField(1,1);
MY(z,1) = m.MagneticField(1,2);
MZ(z,1) = m.MagneticField(1,3);
z = z+1;
x = x+0.2;
end
toc

end
Label(y,1) = 1;
AXmean(y,1) = arrayfun(@(i) mean(AX(i:i+n-1)),1:n:length(AX)-n+1)';
AYmean(y,1) = arrayfun(@(i) mean(AY(i:i+n-1)),1:n:length(AY)-n+1)'; 
AZmean(y,1) = arrayfun(@(i) mean(AZ(i:i+n-1)),1:n:length(AZ)-n+1)'; 
AXstd(y,1) = arrayfun(@(i) std(AX(i:i+n-1)),1:n:length(AX)-n+1)'; 
AYstd(y,1)  = arrayfun(@(i) std(AY(i:i+n-1)),1:n:length(AY)-n+1)'; 
AZstd(y,1) = arrayfun(@(i) std(AZ(i:i+n-1)),1:n:length(AZ)-n+1)'; 
AXmedian(y,1) = arrayfun(@(i) median(AX(i:i+n-1)),1:n:length(AX)-n+1)';
AYmedian(y,1) = arrayfun(@(i) median(AY(i:i+n-1)),1:n:length(AY)-n+1)';
AZmedian(y,1) = arrayfun(@(i) median(AZ(i:i+n-1)),1:n:length(AZ)-n+1)';
G1 = fft(AX);
mean1 = abs(G1);
pow1 = G1.*conj(G1);
G2 = fft(AY);
mean2 = abs(G2);
pow2 = G2.*conj(G2);
G3 = fft(AZ);
mean3 = abs(G3);
pow3 = G3.*conj(G3);        
AXfreqEnergy(y,1) = mean(pow1);
AYfreqEnergy(y,1) = mean(pow2);
AZfreqEnergy(y,1) = mean(pow3);
AXfreqMean(y,1) = mean(mean1); 
AYfreqMean(y,1) = mean(mean2);
AZfreqMean(y,1) = mean(mean3);
AvXmean(y,1) = arrayfun(@(i) mean(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % the averaged vector
AvYmean(y,1) = arrayfun(@(i) mean(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % the averaged vector
AvZmean(y,1) = arrayfun(@(i) mean(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % the averaged vector
AvXstd(y,1) = arrayfun(@(i) std(AvX(i:i+n-1)),1:n:length(AvX)-n+1)'; % std vector
AvYstd(y,1)  = arrayfun(@(i) std(AvY(i:i+n-1)),1:n:length(AvY)-n+1)'; % std vector
AvZstd(y,1) = arrayfun(@(i) std(AvZ(i:i+n-1)),1:n:length(AvZ)-n+1)'; % std vector
MXmean(y,1) = arrayfun(@(i) mean(MX(i:i+n-1)),1:n:length(MX)-n+1)'; % the averaged vector
MYmean(y,1) = arrayfun(@(i) mean(MY(i:i+n-1)),1:n:length(MY)-n+1)'; % the averaged vector
MZmean(y,1) = arrayfun(@(i) mean(MZ(i:i+n-1)),1:n:length(MZ)-n+1)'; % the averaged vector
MXstd(y,1) = arrayfun(@(i) std(MX(i:i+n-1)),1:n:length(MX)-n+1)'; % std vector
MYstd(y,1)  = arrayfun(@(i) std(MY(i:i+n-1)),1:n:length(MY)-n+1)'; % std vector
MZstd(y,1) = arrayfun(@(i) std(MZ(i:i+n-1)),1:n:length(MZ)-n+1)'; % std vector
Test = table(Label, AXmean, AYmean, AZmean, AXstd, AYstd, AZstd, AXmedian, AYmedian, AZmedian, AXfreqEnergy, AYfreqEnergy, AZfreqEnergy, AXfreqMean, AYfreqMean, AZfreqMean, AvXmean, AvYmean, AvZmean, AvXstd, AvYstd, AvZstd, MXmean, MYmean, MZmean, MXstd, MYstd, MZstd);
Predictions = trainedClassifier.predictFcn(Test);
pred(1,1) = Predictions(y,1);
y = y+1;
end
