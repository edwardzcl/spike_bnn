fid1=fopen('input_data_str.txt','wt');
fid2=fopen('input_data.txt','wt');

for chipaddr_y=0:0
    for chipaddr_x=0:0
        for coreaddr_x=0:0
            for coreaddr_y=0:8
                for axonaddr=0:2:12*12*2-1
                    chipaddr_x_bin=dec2bin(chipaddr_x,5);
                    chipaddr_y_bin=dec2bin(chipaddr_y,5);
                    coreaddr_x_bin=dec2bin(coreaddr_x,5);
                    coreaddr_y_bin=dec2bin(coreaddr_y,5);
                    axonaddr_bin=dec2bin(axonaddr,10);
                    input_data_addr_str{coreaddr_y+1}{axonaddr+1}=[chipaddr_x_bin,chipaddr_y_bin,coreaddr_x_bin,coreaddr_y_bin,axonaddr_bin];
                    fprintf(fid1,'%s\n',input_data_addr_str{coreaddr_y+1}{axonaddr+1});
                   
                    input_data_addr{coreaddr_y+1}{axonaddr+1}=bin2dec(input_data_addr_str{coreaddr_y+1}{axonaddr+1});
                    bits=bitget(input_data_addr{coreaddr_y+1}{axonaddr+1},30:-1:1);
                    for i=1:30
                        fprintf(fid2,'%d',bits(i));
                    end
                    fprintf(fid2,'\n');
                end
                fprintf(fid1,'\n');
                fprintf(fid2,'\n');
            end
        end
    end
end

fclose(fid1);
fclose(fid2);