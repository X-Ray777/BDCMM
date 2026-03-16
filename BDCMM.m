 clear;close all;clc;
%% load dataset
load jain
data_with_lable = dataset;
%% deduplicate data 
data_x = unique(data_with_lable,'rows');
if size(data_x,1) ~= size(data_with_lable,1)
    data_with_lable = data_x;
end
lable =  data_with_lable(:,end);
data = data_with_lable(:,1:end-1);
%% BL clustering
fprintf('untitled Clustering :)!\n');
%% normalization
data=(data-min(data))./(max(data)-min(data));
data(isnan(data))=0;
tic;
%% 处理数据
[num,dim]  = size(data);   %数据前两列
mdist=pdist(data);         %两两行之间距离
A=tril(ones(num))-eye(num);%除主对角线的下三角矩阵
[x,y]=find(A~=0);          
% Column 1: id of element i, Column 2: id of element j', Column 3: dist(i,j)'
xx=[x y mdist'];           %点到点的距离矩阵
ND=max(xx(:,2));           %第二列最大值
NL=max(xx(:,1));
if (NL>ND)
  ND=NL; %% 确保 DN 取为第一二列最大值中的较大者，并将其作为数据点总数
end
N=size(xx,1); %% xx 第一个维度的长度，相当于文件的行数（即距离的总个数）

%% 初始化为零
for i=1:ND
  for j=1:ND
    dist(i,j)=0;
  end
end

%% 利用 xx 为 dist 数组赋值，注意输入只存了 0.5*DN(DN-1) 个值，这里将其补成了满矩阵
%% 这里不考虑对角线元素
for i=1:N         %n为距离总个数
  ii=xx(i,1);     %xx第一列的点
  jj=xx(i,2);     %第二列的点
  dist(ii,jj)=xx(i,3);
  dist(jj,ii)=xx(i,3);%给距离矩阵赋值  
end
 
%% 确定 dc
resulthh = zeros(300,5);
xhs = 1;
percent = 2;
fprintf('average percentage of neighbours (hard coded): %5.6f', percent);

position=round(N*percent/100);   %% round 是一个四舍五入函数
sda=sort(xx(:,3));   %% 对所有距离值作升序排列
dc=sda(position);

%% 计算局部密度 rho (利用 Gaussian 核)
fprintf('Computing Rho with gaussian kernel of radius: %12.6f', dc);

%% 将每个数据点的 rho 值初始化为零
for i=1:ND
  rho(i)=0;
end

%     % Gaussian kernel
for i=1:ND-1
  for j=i+1:ND
     rho(i)=rho(i)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
     rho(j)=rho(j)+exp(-(dist(i,j)/dc)*(dist(i,j)/dc));
  end
end
% "Cut off" kernel
%     for i=1:ND-1
%      for j=i+1:ND
%        if (dist(i,j)<dc)
%           rho(i)=rho(i)+1.;
%           rho(j)=rho(j)+1.;
%        end
%      end
%     end
%% 先求矩阵列最大值，再求最大值，最后得到所有距离值中的最大值
maxd=max(max(dist));
%% 将 rho 按降序排列，ordrho 保持序
[rho_sorted,ordrho]=sort(rho,'descend');
%% 处理 rho 值最大的数据点
delta(ordrho(1))=-1.;
nneigh(ordrho(1))=0;
descendant = zeros(ND,1);
%% 生成 delta(i与其密度更高点的最小距离) 和 nneigh 数组
for ii=2:ND
   delta(ordrho(ii))=maxd;
   for jj=1:ii-1
     if(dist(ordrho(ii),ordrho(jj))<delta(ordrho(ii)))
        delta(ordrho(ii))=dist(ordrho(ii),ordrho(jj));
        nneigh(ordrho(ii))=ordrho(jj);
        % 记录 rho 值更大的数据点中与 ordrho(ii) 距离最近的点的编号 ordrho(jj)
     end
   end
end

%% 生成 rho 值最大数据点的 delta 值
delta(ordrho(1))=max(delta(:));
deltamean = mean(delta);
deltastd = std(delta);
%     for wey=0.1:0.1:1
wey=0.5;
deltamin = deltamean+wey*deltastd;
%% 利用 rho 和 delta 画出一个所谓的“决策图”
% tt=plot(rho(:),delta(:),'o','MarkerSize',5,'MarkerFaceColor','k','MarkerEdgeColor','k');
% title ('Decision Graph','FontSize',15.0)
% xlabel ('rho')
% ylabel ('delta')
% yline(deltamin, 'r-', 'LineWidth', 1.5, 'Label', '\delta = deltamin');

%% 初始化 cluster 个数
NCLUST=0;

%% cl 为归属标志数组，cl(i)=j 表示第 i 号数据点归属于第 j 个 cluster
%% 先统一将 cl 初始化为 -1
for i=1:ND
  cl(i)=-1;
end

%% 根据划定的范围统计数据点（即聚类中心）的个数
for i=1:ND
  if ((delta(i)>deltamin))    %if ( (rho(i)>rhomin) && (delta(i)>deltamin))
     NCLUST=NCLUST+1;
     cl(i)=NCLUST;  %% 第 i 号数据点属于第 NCLUST 个 cluster
     icl(NCLUST)=i; %% 逆映射,第 NCLUST 个 cluster 的中心为第 i 号数据点 
  end
end

%% 将其他数据点归类 (assignation)
for i=1:ND
  if (cl(ordrho(i))==-1)
    cl(ordrho(i))=cl(nneigh(ordrho(i)));
  end
end
%% 定义子类稀疏点
for i=1:NCLUST
    cl_edge(i) = length(rho(cl==i)>mean(rho(cl==i)));
end

CL = cl';


%% noise
small_num = 0;
for i = 1:NCLUST
    cl_index = find(cl==i);
    cl_i = data(cl_index,:);
    [m,~] = size(cl_i);
    if m <= 4
        rho_max = max(rho(cl==i));
        qq = cl_index(find(rho(cl_index)==rho_max));
        if qq~=ordrho(1)
        cl(cl_index) = cl(nneigh(min(qq)));
        small_num=small_num+1;
        end
    end
end
nclust = NCLUST - small_num;
null_num = zeros(NCLUST,1);
sum_null = 0;
for i = 1:NCLUST
    cl_i = data(find(cl==i),:);
    if isempty(cl_i)
        null_num(i) = 1;
        sum_null = sum_null+1;
    end
    if find(null_num == 1)
        cl(find(cl==i)) = i - sum_null;
    end
end

%% 计算每个类的平均距离
clk=3;
CL_DIST = zeros(nclust,1);           %初始化类内距离
for i = 1:nclust
    cl_i = data(find(cl==i),:);
    [m,~] = size(cl_i);
    if m == 1
        CL_DIST(i) = 0;
    elseif m == 2
        CL_DIST(i) = pdist(cl_i);
    else
        cl_dist = distfcm(cl_i,cl_i);      %类内任意两点间的距离
        cl_dist = sort(cl_dist,2,'ASCEND');

        col_means = mean(cl_dist, 1);
        CL_DIST(i) = mean(col_means(2:clk));
    end
end

 %% 定义不同类之间距离
CDIST = cell(nclust,nclust);
circle_num = zeros(nclust,nclust);      %小圈个数矩阵 (小圈数是为中心的类的边缘点个数) 
sum_num = 0;
prr=3;
 for i = 1:nclust-1
    for j = i+1:nclust
        xi = data(find(cl==i),:);
        xj = data(find(cl==j),:);
        cutoff = prr*CL_DIST;
        [indij,indij_dist] = rangesearch(xi,xj,cutoff(i));            %根据xj找xi的边缘点
        [n,~]=size(indij);
        count=0;   %统计非空数
        for iii=1:n
            count=count+~isempty(indij{iii,:});
        end
        if (count~=0)          
            indij(cellfun(@isempty,indij)) = [];                   %删除元胞数组中的空白行
            indij_dist(cellfun(@isempty,indij_dist)) = [];
            kk = length(indij);
            circle_num(i,j) = kk;
            sum_num = sum_num+kk; 
            for k = 1:kk
                indij_dist{k} = min(indij_dist{k});  %选取每个小圈内的最小距离
            end
            CDIST{i,j} = reshape(cell2mat(indij_dist),[],1);
            CDIST{j,i} = reshape(cell2mat(indij_dist),[],1);
        end
    end
 end
 hhh=(circle_num~=0);
 num = sum(hhh(:));
 min_num = sum_num/(num);%小圈的平均数的一半
 min_num = floor(sum_num/nclust);
can=3;
 for i = 1:nclust-1
    for j = i+1:nclust
        if circle_num(i,j)< floor(min_num/can)   %小圈数小于最小阈值的两个类
            xi = data(find(cl==i),:);
            xj = data(find(cl==j),:);
            dist_ij = min(distfcm(xi,xj));
            dist_ij = sort([dist_ij],'ascend');
            dist_ij = [dist_ij';zeros(min_num,1)]; 
            dist_ij = dist_ij(1:min_num);
            CDIST{i,j}=dist_ij;            %如果两类没有边界点 距离为两类任意两点间最近值
            CDIST{j,i}=dist_ij;

        end
    end
 end

sim = zeros(nclust,nclust);
sim_list = [];
for pk1=1:nclust-1
    for pk2 =pk1+1:nclust
        smesgs = CDIST(pk1,pk2);
        smesgs = smesgs{:};
        max_smesg = mean(smesgs);
        if circle_num(pk1,pk2) <= floor(min_num/can)
            sim(pk1,pk2) = 1;
        end
         if circle_num(pk1,pk2) > floor(min_num/can)
              sim(pk1,pk2) = max_smesg/circle_num(pk1,pk2);
        end
        sim_list = [sim_list sim(pk1,pk2)];
    end
end

 %% Single-linkage clustering of sub-clusters according to SIM
SingleLink = linkage(sim_list,'single');
bata = [0;SingleLink(:,end)];
bata(bata<0)=0;
bataratio = [[nclust+1-(1:nclust-1)]' diff(bata)];
bataratio = sortrows(bataratio,2,'descend');
NC = bataratio(1,1); %% the stable number of cluster that with maximum bata-interval.
F_sub_L = cluster(SingleLink,NC); 
for i=1:nclust
    AA = find(cl==i);
    CL(AA) = F_sub_L(i); %% CL: the cluster label
end
for i=1:2
    nccl=find(CL==i);
    [BB,cc] = max(rho(nccl));
    icl(i) = cc;    
end
runtime = toc
fprintf('Finished!!!!\n'); 
 %% evaluation
lable=lable+1;
[AMI,ARI,FMI] = Evaluation(CL,lable)
NMI=nmi(CL,lable)