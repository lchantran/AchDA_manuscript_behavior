clear
% root = '/Users/lynnechantranupong/Dropbox (HMS)/2ABT_Celia/Vglut2 tetanus';
root = '/Users/lynnechantranupong/Dropbox (HMS)/Lynne_behavior_transfer5/behavior/';
% root = '/Volumes/Neurobio/MICROSCOPE/Lynne/2ABT/test';
% root = '/Volumes/Neurobio-1/MICROSCOPE/Lynne/2ABT/test';

cd(root)
dates = dir;
for date=4:length(dates)
    cd(dates(date).name)
    mice = dir;    
    for mouse=1:length(mice)
        if isempty(strfind(mice(mouse).name, '.')) && isempty(strfind(mice(mouse).name, 'err'))
            cd(mice(mouse).name)
      
            if length(dir)>4
                
                try
                    temp = ls('*.csv');
                catch 
                    temp = '';
                end

                 % only load in data if _trials.csv doesn't already exist:
                 
                if isempty(findstr('parameters.csv', temp)) && size(dir('*.mat'),1)~=0 % check that .mat files exist
                    matFiles = dir('*.mat');
                    
                        %loads the stats, pokeHistory, and parameters
                    for iFile = 1:size(matFiles,1)
                        load(matFiles(iFile).name);
                    end
                    
                    try
                        trials = extractTrials_opto_center(stats, pokeHistory); 
                        % extractTrials_opto_for_csv - pre Jan 2022
                        % extractTrials_opto_center - ofer new code: Jan
                        % 2022 onwards
                        trial_filename = [mice(mouse).name, '_', dates(date).name, '_trials.csv'];
                        csvwrite(trial_filename, trials);
                        p_filename = [mice(mouse).name,'_',dates(date).name, '_parameters','.csv'];
                        writetable(struct2table(p),p_filename);
                        %print(trial_filename)
                    catch ME
                        if ME.stack(1).line == 78
                            warning('no Trials completed for %s on %s', mice(mouse).name, dates(date).name);
                        else
                            warning('unidentified problem with trial extraction for %s on %s', mice(mouse).name, dates(date).name)
                        end
                    end
                end
            else warning('No files found for %s on %s', mice(mouse).name, dates(date).name)
            end
            cd ..
        end
        clear temp pokeHistory stats trial_filename p_filename trials
    end
    cd ..
end 

load chirp
sound(y,Fs)