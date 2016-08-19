% the smaller the output number is, the more obvious the difference is
function j = Jfeatures(Ra,datasize)
    d = datasize(1);                % size of features
    a = datasize(2);                % size of action
    [~,iteration] = size(Ra);
    [folder,~] = size(Ra{1});
    sum = 0;
    for i = 1:iteration
        for j = 1:folder
            group = nchoosek(1:a,2);
            times = size(group,1);
            temp = 0;
            for k = 1:times
                added = length(intersect(Ra{i}{j}{group(k,1)},Ra{i}{j}{group(k,2)}))...
                    /length(union(Ra{i}{j}{group(k,1)},Ra{i}{j}{group(k,2)}));
                if ~isnan(added)
                    temp = temp + added;
                else
                    temp = temp + 1;
                end
            end
            sum = sum + 2*temp/times;
        end
    end
    j = sum/(d*a);
end