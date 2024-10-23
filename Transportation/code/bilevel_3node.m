

%% Inicializacion red 3 nodos

clear all;
close all;
clc;
cvx_begin quiet
cvx_end
n = 3;


d = 2.*(ones(n) - eye(n));
u = 3.*(ones(n) - eye(n));

c = ones(n) + 10.*eye(n);
m = 2.*(ones(n) - eye(n));
a = 3.*(ones(n) - eye(n));

d = 2.*(ones(n) - eye(n));

d = [0,1,10;1,0,2;10,2,0];

u = 3.*(ones(n) - eye(n));

c = ones(n) + 10.*eye(n);
c = [0,1,10;1,0,3;10,3,0];


m = 2.*(ones(n) - eye(n));
m = [0,2,6;2,0,1;6,1,0];



%% Check computational times for different initializations for inner problem




comp_time = zeros(10,1);
max_iters = 1.2e6;
obj_vecs = zeros(10,max_iters);
obj_vecs_f = zeros(10,max_iters);
loss_vec = zeros(10,max_iters);
yloss_vec = zeros(10,max_iters);
gamma = 4;
fg_vec = zeros(10,n,n);
ff_vec = fg_vec;

%rng(1);


for i=1:10
    gamma = i;
    rng(1);
    a_start = 0.7.*rand(n,n) + 0.3;
    tic;
    beta = 3.2e-6;
    [a,fg,ff,fijg,lamf,muf,a_vec,f_vec,fij_vec,lamf_vec,muf_vec,obj_vec,obj_vec_f,loss,yloss] = A3_ods(n,d,u,gamma,m,c,a_start,beta);
    obj_vecs(i,1:length(obj_vec)) = obj_vec;
    obj_vecs_f(i,1:length(obj_vec_f)) = obj_vec_f;
    loss_vec(i,:) = loss;
    yloss_vec(i,:) = yloss;
    fg_vec(i,:,:) = fg;
    ff_vec(i,:,:) = ff;
    % obj_vecs(i,length(obj_vec)+1:end) = max(obj_vec).*ones(1,length(obj_vecs)-length(obj_vec));
    comp_time(i) = toc;
    disp(['i = ',num2str(i),', comp_time = ',num2str(comp_time(i))]);
end

%%
close all;
figure;
dif_norm = zeros(1,10);
for i=1:10
    dif_norm(i) = sum(sum(abs(fg_vec(i,:) - ff_vec(i,:))));
end
subplot(211);
plot(dif_norm);
xlabel('$\gamma$','Interpreter','latex');
ylabel('|y_F-y^*|');
colors  = [
    0.00, 0.45, 0.74;  % Blue
    0.85, 0.33, 0.10;  % Red
    0.93, 0.69, 0.13;  % Yellow
    0.49, 0.18, 0.56;  % Purple
    0.47, 0.67, 0.19;  % Green
    0.30, 0.75, 0.93;  % Cyan
    0.64, 0.08, 0.18;  % Dark Red
    0.75, 0.75, 0.00;  % Olive
    0.35, 0.35, 0.35;  % Dark Gray
    0.10, 0.75, 0.30;  % Bright Green
];
subplot(212);

for i=1:10
    time = linspace(0,comp_time(i),length(obj_vecs_f(i,:)));
    plot(time,obj_vecs(i,:),'LineWidth',1.5,'Color', colors(i, :));
    hold on;
    plot(time,obj_vecs_f(i,:),'--','LineWidth',1.5,'Color', colors(i, :));
    hold on;
end
legend('$f(x,y_g),\gamma = 1$','$f(x,y_F),\gamma = 1$','$f(x,y_g),\gamma = 2$', ...
    '$f(x,y_F),\gamma = 2$','$f(x,y_g),\gamma = 3$', ...
    '$f(x,y_F),\gamma = 3$', ...
    '$f(x,y_g),\gamma = 4$','$f(x,y_F),\gamma = 4$', ...
    '$f(x,y_g),\gamma = 5$','$f(x,y_F),\gamma = 5$', ...
    '$f(x,y_g),\gamma = 6$','$f(x,y_F),\gamma = 6$',...
    '$f(x,y_g),\gamma = 7$','$f(x,y_F),\gamma = 7$', ...
    '$f(x,y_g),\gamma = 8$','$f(x,y_F),\gamma = 8$', ...
    '$f(x,y_g),\gamma = 9$','$f(x,y_F),\gamma = 9$',...
    '$f(x,y_g),\gamma = 10$','$f(x,y_F),\gamma = 10$',...
    'Interpreter','latex');
xlabel('time [s]','Interpreter','latex'); ylabel('$-f(x,y)$','Interpreter','latex');
%% calculate comp_time

for i=1:10
    for k=100:1000:length(obj_vecs(i,:))

        if (abs(obj_vecs(i,k) - obj_vecs(i,max(k-1000,1)) ) < 1e-3) && (abs(obj_vecs(i,k) - obj_vecs(i,min(k+500,end)) ) < 1e-3) && (abs(obj_vecs(i,k) - obj_vecs(i,min(k+1000,end)) ) < 1e-3) && (abs(obj_vecs(i,k) - obj_vecs(i,end) ) < 1e-3)
            %disp(i)
            %disp(obj_vecs(i,end));
            disp(comp_time(i).*k./length(obj_vecs(i,:)));
            break;
        end

    end
end
%%

%%
% %% Results representation
% 
% 
% close all;
% 
% map_x = [2,6,6,11,11,13,14,14,1];
% map_y = [4,6,2,6,2,4,1,7,1];
% 
% 
% figure(1);
% for i=1:length(x)
%     subplot(2,2,i);
%    % imagesc(a_vecs(:,:,i));
%    a = a_vecs(:,:,i);
%    a(a < 0.01) = 0;
%    a = a  + a';
%    g = graph(a);
%    % h = plot(g);
% 
%     h = plot(g,'XData',map_x,'YData',map_y,'LineWidth',0.7.*g.Edges.Weight.^0.7,'NodeFontSize',5,...
%     'EdgeColor','#0072BD','EdgeAlpha',0.8,'interpreter','latex');%, ...
%          %   'MarkerSize',0.7*(s_h+s).^0.5 +1e-2);%,'LineWidth', ...
%             %0.1.*g.Edges.Weight,'NodeColor',colores,'EdgeColor',colores_edg,'EdgeAlpha',0.7,'NodeFontSize',8);
%         xticks([]); yticks([]); 
% 
%    tit = sprintf('$ \\gamma = %d $, $ A_0 =  %.2f$, obj = $ %.2f$, t = $ %.2f$',x(i),1,max(obj_vecs(i,2:end)),comp_time(i));
%    title(tit,'Interpreter','latex','FontSize',5);
% 
%    % title(tit,'FontSize',8);
% end
% 
% figure(2);
% for i=1:length(x)
%     subplot(2,2,i);
%     imagesc(f_vecs(:,:,i));
%     colorbar;
%     hold on;
%     for ii = 1:n
%         for jj = 1:n
%             text(jj, ii, num2str(f_vecs(ii, jj,i)), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'white');
%             hold on;
%         end
%     end
% end





% 
% %% Lower level for different configurations of the network
% 
% close all;
% xx = 0:0.01:1; x_x = 1-xx;
% y1 = xx.*(log(xx) - 1) + x_x.*(log(x_x) - 1) + u.*(x_x) + (d(1,2) + d(2,4) + d(4,8)).*xx;
% xx = fg_vec(1); x_x = 1-xx;
% y1_op = xx.*(log(xx) - 1) + x_x.*(log(x_x) - 1) + u.*(x_x) + (d(1,2) + d(2,4) + d(4,8)).*xx;
% 
% 
% 
% xx = 0:0.01:1; x_x = 1-xx;
% y2 = xx.*(log(xx) - 1) + x_x.*(log(x_x) - 1) + u.*(x_x) + (d(1,3) + d(3,4) + d(4,8)).*xx;
% xx = fg_vec(2); x_x = 1-xx;
% y2_op = xx.*(log(xx) - 1) + x_x.*(log(x_x) - 1) + u.*(x_x) + (d(1,3) + d(3,4) + d(4,8)).*xx;
% 
% 
% xx = 0:0.01:1; x_x = 1-xx;
% y3 = xx.*(log(xx) - 1) + x_x.*(log(x_x) - 1) + u.*(x_x) + (d(1,3) + d(3,5) + d(5,6) + d(6,8)).*xx;
% xx = fg_vec(3); x_x = 1-xx;
% y3_op = xx.*(log(xx) - 1) + x_x.*(log(x_x) - 1) + u.*(x_x) + (d(1,3) + d(3,5) + d(5,6) + d(6,8)).*xx;
% 
% 
% xx = 0:0.01:1; x_x = 1-xx;
% subplot(311); 
% plot(xx,y1); hold on; scatter(fg_vec(1),y1_op,'red','filled'); xlabel('F'); ylabel('PAX(A,F)'); title('A_0 = 0.1')
% subplot(312); plot(xx,y2); hold on; scatter(fg_vec(2),y2_op,'red','filled'); xlabel('F'); ylabel('PAX(A,F)'); title('A_0 = 0.6')
% subplot(313); plot(xx,y3); hold on; scatter(fg_vec(3),y3_op,'red','filled'); xlabel('F');ylabel('PAX(A,F)'); title('A_0 = 1')



%% Functions - all ods



function [f,fij,lam,mu] = A2_ods(n,d,u,a,f_last,fij_last,lam_last,mu_last)
    lam = lam_last;
    f = f_last;
    fij = fij_last;
    mu = mu_last;

    mu_o = zeros(n);
    mu_d = zeros(n);

    mu_i = zeros(n,n,n,n);
    mu_j = mu_i;

    lam_prev = -10*ones(n); %i,j
    f_prev = -10*ones(n); %od
    fij_prev = -10*ones(n,n,n,n); %i,j,o,d
    mu_prev = -10*ones(n,n,n); %i,o,d

    beta_1 = 1e-2;
    beta_2 = 4e-1;
    q = 0;
    max_dif = 1;

    while ( mean(mean(abs(f-f_prev)))/(beta_1) > 2e-2 ) ...
       || ( mean(mean(mean(mean(abs(fij-fij_prev)))))/(beta_1) > 1e-2 ) ...
       || (min(min((squeeze(sum(sum(fij,4),3)) < (a+1e-2)))) < 0.5) ...
       || (max_dif > 2e-1)

        lam_prev = lam;
        f_prev = f;
        fij_prev  = fij;
        mu_prev = mu;
   
     %   disp(max_dif)

       

       mu = mu + beta_2.*(squeeze(sum( fij , 2 ))  - ...
          squeeze(permute(sum(fij , 1 ),[2 1 3 4])));

%         for o=1:n
%             for des=2
%                 mu(:,o,des) = mu(:,o,des) + beta*(sum( fij(:,:,o,des),2 ) - sum(fij(:,:,o,des) , 1)' );
%             end
%         end

        for o=1:n
            for des=1:n
                mu(o,o,des) = mu(o,o,des) + beta_2*(-f(o,des));
                mu_o(o,des) = mu(o,o,des);
                mu(des,o,des) = mu(des,o,des) + beta_2*(f(o,des));
                mu_d(o,des) = mu(des,o,des);
            end
        end

     %   mu = max(mu,0);

        for o=1:n
            for des=1:n
                mu_i(:,:,o,des) = mu(:,o,des)*ones(1,n);
                mu_j(:,:,o,des) = permute(mu_i(:,:,o,des),[2,1,3,4]);
            end
        end

        for i=1:n
            for j=1:n %cambiar con los candidatos
                lam(i,j) = lam(i,j) + beta_2.*(   sum(sum(squeeze(permute(fij(i,j,:,:),[3,4,1,2])))) - a(i,j) );
            end
        end

%         for i=1:n
%             for j=1:n
%                 lam(i,j) = lam(i,j) + beta*(  sum(sum(squeeze(fij(i,j,:,:)))) - a(i,j));
%             end
%         end

        lam = max(0,lam);

        for i=1:n
            lam(i,i) = 100;
        end

        f = f - beta_1*(log(f./(1-f))  - u - mu_o + mu_d );

%         for o=1:n
%             for des=1:n
%                 f(o,des) = f(o,des) - beta*(log(f(o,des)./(1-f(o,des)))  - u(o,des) - mu(o,o,des) + mu(des,o,des) ); %puedo quitar algun bucle?
%             end
%         end

        
        f = min(0.99,f);
        f = max(0.01,f);

        for o=1:n
            for des=1:n
                fij(:,:,o,des) = fij(:,:,o,des) - beta_1*( d + lam + mu_i(:,:,o,des) - mu_j(:,:,o,des) );
            end
        end

%         for o=1:n
%             for des=1:n
%                 for i=1:n
%                     for j=1:n
%                         fij(i,j,o,des) = fij(i,j,o,des) - beta*(d(i,j) + lam(i,j) + mu(i,o,des) - mu(j,o,des)  );
%                     end
%                 end
%                % fij(:,:,o,des) = fij(:,:,o,des) - beta*(d + lam + mu_i - mu_j);
%             end
%         end

        fij = min(1,fij);
        fij = max(0,fij);

        for i=1:n
            f(i,i) = 0;
            fij(:,:,i,i) = 0;
            mu(:,i,i) = 0;
        end

        cons_f_a = 0;
        for i=1:n
            for j=1:n %cambiar con los candidatos
                if squeeze(sum(sum(squeeze(permute(fij(i,j,:,:),[3 4 1 2]))))) >= (a(i,j) + 3e-1)
                    cons_f_a = cons_f_a + 1;
                end
            end
        end



        max_dif = 0;
        
        for oo=1:n
            for dd=1:n
                if (oo ~= dd)
                    max_dif = max(max_dif, abs(sum(fij(oo,:,oo,dd),2) - f(oo,dd))  );
                end
                
            end
        end


        if q == 100
%             disp('fij 1,2 = ');
%             disp(fij(:,:,1,2));
%             disp('f 1,2 = ');
            %disp(f);
            q = 0;
        else
            q = q+1;
        end
        
        
    end

end

function [f,fij,lam,mu] = A2_f_ods(n,d,u,a,gamma,m,f_last,fij_last,lam_last,mu_last)
    
    lam = lam_last;
    f = f_last; 
    fij = fij_last;
    mu = mu_last;

    lam_prev = -10*ones(n);
    f_prev = -10*ones(n);
    fij_prev = -10*ones(n,n,n,n);
    mu_prev = -10*ones(n,n,n);

    beta_1 = 1e-2/gamma; %gamma 10
    beta_2 = 4e-1/gamma; %gamma 10

    max_dif = 1;
    cons_f_a  = 20;
   

    while ( mean(mean(abs(f-f_prev)))/(beta_1*gamma) > 3e-2 ) ...
       || (mean(mean((squeeze(sum(sum(fij,4),3)) < (a+1e-2)))) < 0.5) ...
       || (max_dif > 3e-1)  ...
          || ( mean(mean(max(max(abs(fij-fij_prev)))))/(beta_1*gamma) > 1e-1 )
        %



        lam_prev = lam;
        f_prev =f;
        fij_prev = fij;
        mu_prev = mu;

       % disp(max_dif)
       %disp(['fij 1-2 = ',num2str(max(max(fij(:,:,1,2)))),', f 1-2 = ',num2str(f(1,2))]);
    

        mu = mu + beta_2.*(squeeze(sum( fij , 2 ))  - ...
          squeeze(permute(sum(fij , 1 ),[2 1 3 4])));

        for o=1:n
            for des=1:n
                mu(o,o,des) = mu(o,o,des) + beta_2.*(-f(o,des));
                mu_o(o,des) = mu(o,o,des);
                mu(des,o,des) = mu(des,o,des) + beta_2.*(f(o,des));
                mu_d(o,des) = mu(des,o,des);
            end
        end

      %  mu = max(mu,0);

        for o=1:n
            for des=1:n
                mu_i(:,:,o,des) = mu(:,o,des)*ones(1,n);
                mu_j(:,:,o,des) = permute(mu_i(:,:,o,des),[2,1,3,4]);
            end
        end

        for i=1:n
            for j=1:n %cambiar a candidatos
                lam(i,j) = lam(i,j) + beta_2.*(   sum(sum(squeeze(permute(fij(i,j,:,:),[3,4,1,2])))) - a(i,j) );
            end
        end
        lam = max(lam,0);

        for i=1:n
            lam(i,i) = 100;
        end

        f = f - beta_1*(-m  + gamma.*(log(f./(1-f))  - u) - mu_o + mu_d );
        f = min(0.99,f);
        f = max(0.01,f);



        for o=1:n
            for des=1:n
                fij(:,:,o,des) = fij(:,:,o,des) - beta_1*( gamma.*d + mu_i(:,:,o,des) - mu_j(:,:,o,des) + lam);
            end
        end

        fij = min(1,fij);
        fij = max(0,fij);

        cons_f_a = 0;
        for i=1:n
            for j=1:n %cambiar a candidatos
                if squeeze(sum(sum(squeeze(permute(fij(i,j,:,:),[3 4 1 2]))))) >= (a(i,j) + 3e-1)
                    cons_f_a = cons_f_a + 1;
                end
            end
        end

        max_dif = 0;
        for oo=1:n
            for dd=1:n
                if (oo ~= dd)
                    max_dif = max(max_dif, abs(sum(fij(oo,:,oo,dd),2) - f(oo,dd))  );

                end
                
            end
        end
      %  disp(dif_ok./total_od);
       % disp(max_dif);

        for i=1:n
            f(i,i) = 0;
            fij(:,:,i,i) = 0;
            mu(:,i,i) = 0;
        end

    end

end




function [a,fg,ff,fijg_best,lamf_best,muf_best,a_vec,f_vec,fij_vec,lamf_vec,muf_vec,obj_vec,obj_vec_f,loss_vec,yloss] = A3_ods(n,d,u,gamma,m,c,a_start,beta_base)

    a = a_start;

    fg = 0.01*ones(n);
    fijg = zeros(n,n,n,n);
    lamg = zeros(n);
    mug = zeros(n,n,n);

    ff = fg;
    fijf = fijg;
    lamf = lamg;
    muf = mug;

    a_prev = -10*ones(n);
    f_prev = -10*ones(n);
    fij_prev = -10*ones(n,n,n,n);
    lam_prev = -10*ones(n);
    mu_prev = -10*ones(n,n,n);

    % beta = 1e-4; %gamma 10
    % beta = 1e-5;
    % beta = 1e-6;
    
    beta = beta_base;
    loss_vec = [];
    yloss = [];

    numdatos = 3e2;
    obj_vec_lasts_g = zeros(1,numdatos);
    obj_vec_lasts_f = zeros(1,numdatos);
    
    hplot = plot(1:numdatos,obj_vec_lasts_g,'-b',1:numdatos,obj_vec_lasts_f,'--b','LineWidth',1.5);
    xlabel('Last 300 results');
    ylabel('F(a,f(a))');
    ylim([-10,inf]);
    %legend('F(a,f)','Obj(a,\theta)','Location','northwest');



    a_vec = [a];
    f_vec = [fg];
    fij_vec = [fijg];
    lamf_vec = [lamf];
   % lamg_vec = [lamg];
    muf_vec = [muf];
   % mug_vec = [mug];
    obj_vec = [];
    obj_vec_f = [];
    lamg_prev = -10*ones(n);
    max_iters = 1.2e6;
    obj_best = -1e6;
    k = 0;
    q = 0;
    control = 0;

    % while ((  max(max(abs(fg - f_prev))) > 1e-2) || ( sum(sum((abs(a-a_prev)./a_prev)/beta))  > 1) || ...
    %       (max(max(max(max(abs(fijg-fij_prev))))) > 1e-4) || ...
    %       (max(max(abs(lamf-lam_prev)))/beta > 1e-4) ||...
    %       (max(max(max(abs(muf-mu_prev))))/beta > 1e-2)) && ((k < max_iters)) %  ||  (obj <= (max(obj_vec) - 0.5 )) ) && (k < (max_iters + 2e4))
    % 
    %     %disp(['a gap = ',num2str( sum(sum((abs(a-a_prev)./a_prev)/beta)) )]);
    while (k < max_iters)

        
     %   disp( (abs(a-a_prev)./a_prev) );

     
        a_prev = a;
        f_prev = fg;
        ff_prev = ff;
        fij_prev = fijg;
        fijf_prev = fijf;
        lam_prev = lamf;
        lamg_prev = lamg;
        mu_prev = muf;
        mug_prev = mug;

        %disp(lamg);

        if (max(max(lamg_prev - 1000*eye(n))) < 1) && (k > 1)
            beta = 1e-3;
          %  disp('acelero');
        else
            % beta = 1e-4; %gamma 10
            % beta = 1e-5;
            % beta = 1e-6;
            beta = beta_base;
     
        end
       

        [fg,fijg,lamg,mug] = A2_ods(n,d,u,a,f_prev,fij_prev,lamg_prev,mug_prev);
       % disp('converge el lower');
        [ff,fijf,lamf,muf] = A2_f_ods(n,d,u,a,gamma,m,ff_prev,fijf_prev,lam_prev,mu_prev);
        gt = c + gamma.*lamg - lamf;
        a = a - beta.*gt;

        ffg = fg;
        ffg(ffg < 3e-2) = 0;
        ffijg = fijg;
        ffijg(ffijg < 3e-2) = 0;


        fff = ff;
        fff(fff < 3e-2) = 0;
        ffijf = fijf;
        ffijf(ffijf < 3e-2) = 0;;


        ob_g = -sum(sum(entr(ffg))) - sum(sum(ffg)) - sum(sum(entr(1-ffg))) - sum(sum(1-ffg));
        for oo=1:n
            for dd=1:n
                ob_g = ob_g + u(oo,dd)*(1-ffg(oo,dd));
                for i=1:n
                    for j=1:n
                        ob_g = ob_g + d(i,j)*ffijg(i,j,oo,dd);
                    end
                end
            end
        end

        ob_f = -sum(sum(entr(fff))) - sum(sum(fff)) - sum(sum(entr(1-fff))) - sum(sum(1-fff));
        for oo=1:n
            for dd=1:n
                ob_f = ob_f + u(oo,dd)*(1-fff(oo,dd));
                for i=1:n
                    for j=1:n
                        ob_f = ob_f + d(i,j)*ffijf(i,j,oo,dd);
                    end
                end
            end
        end
        

        
        loss = ob_f - ob_g;
        dif_y = sum(sum(abs(ffg-fff).^2)) + sum(sum(sum(sum(abs(ffijg-ffijf).^2))));
        n2_y = sum(sum(ffg.^2)) + sum(sum(sum(sum(ffijg.^2))));
        yloss = [yloss dif_y/n2_y];
        loss_vec = [loss_vec loss];


         %  disp(['dlamg/dx = '   ,num2str(max(max(abs(lamg-lamg_prev)./abs(a-a_prev))))]);

        a = max(0.001,a);
      %  a = min(0.999,a);
      %  a_vec = [a_vec a(1,3)];
      %  f_vec = [f_vec fg];

      aa_ob = a;
      aa_ob(aa_ob < 1e-2) = 0;
      ff_ob = fg;
      ff_ob_f = ff;
      ff_ob(ff_ob < 2e-2) = 0;
      ff_ob_f(ff_ob_f < 2e-2) = 0;

        obj = sum(sum(m.*ff_ob)) - sum(sum(aa_ob.*c));
        obj_f = sum(sum(m.*ff_ob_f)) - sum(sum(aa_ob.*c));


    
            obj_vec = [obj_vec obj];
            obj_vec_f = [obj_vec_f obj_f];


     


            if (max(obj_vec) == obj)
                a_best = a;
                fg_best = fg;
                fijg_best = fijg;
                lamf_best = lamf;
                muf_best = muf;
            end


      %  lamf_vec = [lamf_vec lamf];
      %  lamg_vec = [lamg_vec lamg];
      %  muf_vec = [muf_vec muf];
     %   mug_vec = [mug_vec mug];

        % if (beta == 1e-2)
        %     k = k + 100;
        %     q = q + 100;
        %     control = control + 100;
        % else
            k = k+1;
            q = q+1;
            control = control + 1;
        % end
    %    if q >= 20000 gamma 10
         if (q >= 20000)  %gamma 20 
%             disp('fijg = ');
%             disp(fijg);
%             disp('fijf = ');
%             disp(fijf);
            disp('fg = ');
            disp(fg);
            disp('a = ');
            disp(a);
            q = 0;


            % if ( max( obj_vec ) <= (obj_best*1.005) )% && (k < max_iters))
            %      k = max_iters;
            % end
            obj_best = max(obj_vec);
            disp(['k = ',num2str(k),', best obj = ',num2str(obj_best)]);
        end

        if control >= 1e3
            disp(['k = ',num2str(k),', obj = ',num2str(obj),', beta = ',num2str(beta)]);
            ares = a;
            ares(ares < 0.1) = 0;
            fgres = fg;
            ffres = ff;
            fgres(fgres<0.02) = 0;
            ffres(ffres<0.02) = 0;
            res_g = sum(sum(m.*fgres))-sum(sum(c.*ares));
            res_f = sum(sum(m.*ffres))-sum(sum(c.*ares));
            obj_vec_lasts_g = [obj_vec_lasts_g(2:end),res_g];
            obj_vec_lasts_f = [obj_vec_lasts_f(2:end),res_f];

            set(hplot(1),'YData',obj_vec_lasts_g);
            set(hplot(2),'YData',obj_vec_lasts_f);
            drawnow;
           % disp(a);
           % disp(max(max(lamg-1000.*eye(n))));
            control = 0;
        end
        
       % disp(['k = ',num2str(k),', x = ',num2str(x),', y = ',num2str(yg), ', lam = ',num2str(lamf) , ', lam_op = ',num2str(lamg)]);
    end
   % disp(a);
   % disp(fg);
   % disp(fijg);
    %disp(['a = ',num2str(x),', y = ',num2str(yg), ', lam = ',num2str(lamf) , ', lam_op = ',num2str(lamg)]);
end
