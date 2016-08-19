% 
function output = LEAR_ratio(Ra,datasize)
    d = datasize(1);                % size of features
    a = datasize(2);                % size of action
    [~,iteration] = size(Ra);
    [folder,~] = size(Ra{1});
    sum1 = 0;
    for i = 1:iteration
        for j = 1:folder
            temp1 = 0;
            for k = 1:a
                temp1 = temp1 + length(Ra{1,i}{j}{k});
            end
            sum1 = sum1 + temp1/(d*a);
        end
    end
    output = sum1/(iteration*folder);
end