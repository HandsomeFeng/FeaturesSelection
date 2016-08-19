clear;
load('r_vehicle');
load('r_pendigits');
load('r_satimage');
load('r_optdigits');

acc = [(acc_vehicle(2:3)-acc_vehicle(4))./acc_vehicle(4);
    (acc_pendigits(2:3)-acc_pendigits(4))./acc_pendigits(4);
    (acc_satimage(2:3)-acc_satimage(4))./acc_satimage(4);
    (acc_optdigits(2:3)-acc_optdigits(4))./acc_optdigits(4)];

ratio = zeros(4,2);
ratio(1,:) = [LEAR_ratio(Ra_vehicle(1,:),size_vehicle),...
    LEAR_ratio(Ra_vehicle(2,:),size_vehicle)];
ratio(2,:) = [LEAR_ratio(Ra_pendigits(1,:),size_pendigits),...
    LEAR_ratio(Ra_pendigits(2,:),size_pendigits)];
ratio(3,:) = [LEAR_ratio(Ra_satimage(1,:),size_satimage),...
    LEAR_ratio(Ra_satimage(2,:),size_satimage)];
ratio(4,:) = [LEAR_ratio(Ra_optdigits(1,:),size_optdigits),...
    LEAR_ratio(Ra_optdigits(2,:),size_optdigits)];

figure;
% bar(ratio);
bar(1:4,[ratio(:,1) acc(:,1)]);
set(gca,'XTickLabel',{'vehicle','pendigits','satimage','optdigits'});
xlabel('dataset');
ylabel('Rate');
legend('Rate','gain accuracy');
axis([0.5,4.5,0,1]);
% title('selected features over all features');

figure;
% bar(ratio);
bar(1:4,[ratio(:,2) acc(:,2)]);
set(gca,'XTickLabel',{'vehicle','pendigits','satimage','optdigits'});
xlabel('dataset');
ylabel('Rate');
legend('Rate','gain accuracy');
axis([0.5,4.5,0,1]);
% title('selected features over all features');

j = zeros(4,2);
j(1,:) = [Jfeatures(Ra_vehicle(1,:),size_vehicle),...
    Jfeatures(Ra_vehicle(2,:),size_vehicle)];
j(2,:) = [Jfeatures(Ra_pendigits(1,:),size_pendigits),...
    Jfeatures(Ra_pendigits(2,:),size_pendigits)];
j(3,:) = [Jfeatures(Ra_satimage(1,:),size_satimage),...
    Jfeatures(Ra_satimage(2,:),size_satimage)];
j(4,:) = [Jfeatures(Ra_optdigits(1,:),size_optdigits),...
    Jfeatures(Ra_optdigits(2,:),size_optdigits)];

figure;
% bar(ratio);
bar(1:4,[j(:,1) acc(:,1)]);
set(gca,'XTickLabel',{'vehicle','pendigits','satimage','optdigits'});
xlabel('dataset');
ylabel('Javg');
legend('Javg','gain accuracy');
axis([0.5,4.5,0,0.5]);
% title('intersection over union');

figure;
% bar(ratio);
bar(1:4,[j(:,2) acc(:,2)]);
set(gca,'XTickLabel',{'vehicle','pendigits','satimage','optdigits'});
xlabel('dataset');
ylabel('Javg');
legend('Javg','gain accuracy');
axis([0.5,4.5,0,0.3]);
% title('intersection over union');