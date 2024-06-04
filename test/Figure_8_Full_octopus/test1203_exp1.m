highlightIndices1 = [ 307,  556,  575,   53,  148,  414,   57,  956,   55,  339,  628,...
        971,  574,  909,    1,  602,  463, 1001,    0,   69,  358,  992,...
        218,   96,  906,  999,  582,  623,  914,  687,  621,  800,  354,...
        994,  831,  836,  616,  993,  352,  686,  834,  349,   81,  742,...
        997,  752,  608,   83,  547,  784,  987,  667,  362,  392,  249,...
        809,  984,  400,  629,  265,  102,  883,  716,  258,  355,  678,...
        160,  100,  581,  782,  931,  254,  459,  775,  882,  618,  756,...
        939,  159,  351,  700,  761,    7,  693,  759,  748,  416,  731,...
        268,  270,  136,  708,  818,  174,   17,  523,  263,  721,  703,...
        106,  634,  288,  114,  255,  165,  509,  198,   38,  367,  712,...
        276,   28,  105,  550,  475,  266,  727,  157,  172,  298,  156,...
        273,  668,  389,  421,  378,  747,  181,  204,  112,  554,  501,...
         33,  173,  935,  898,  948,   64,  597,  968,  947,  944,  328,...
        539]+1;

highlightIndices2 = [925, 841, 843, 817, 818, 776, 773, 753, 684, 682, 983, 614, 957,...
       579, 448, 934, 208, 510, 874, 169, 421, 587, 889, 106, 419, 835,...
       105, 852, 882, 105, 794, 125, 770, 881, 124, 418, 699, 168, 480,...
       416, 976, 424, 356, 314, 381, 955, 230, 404, 182, 790, 812, 166,...
       379, 164, 162, 458, 853, 177, 437, 250, 375, 655, 830, 244, 238,...
       242, 805, 868, 192, 378, 865, 189, 220, 821, 133, 240, 849, 115,...
       139, 333, 810, 119, 272,  56,  99, 436,  63,  78, 200, 468,  55,...
        71, 273, 493,  61, 159, 465,  60, 143, 492,  59, 123, 528,  25,...
        39,   8,  30,   1,   0,  20,   6,  21,  16,  12,  11,   2,   4,...
         0,   0,   0,   0,   0,   0,   0,   0,   0,   0]+1;

%%%%%%%%%%%%%%
%modifyAndSaveOFF1('/Users/salovjade/Desktop/sgw_octopus/octopus1.off', '/Users/salovjade/Desktop/sgw_octopus/modioctopus1.off');


%modifyAndSaveOFF2('/Users/salovjade/Desktop/sgw_octopus/octopus2.off', '/Users/salovjade/Desktop/sgw_octopus/modioctopus2.off');

filenames = {'/Users/salovjade/Desktop/sgw_octopus/modioctopus1.off', '/Users/salovjade/Desktop/sgw_octopus/modioctopus2.off'};
fileColors = {0.3*[1,1,1], 0.3*[1,1,1]};

% Define your file names (or thresholds)
% thresholds = [0.2, 1.5, 2.0];

fileBaseName = 'highlight_indices_';
% Create a figure for plotting
figure;
% sgtitle('3D Octopus Matching');

% First, create the large subplot for the 1.5 case
% Manually position this subplot
% posLarge = [0.1, 0.5, 0.8, 0.4]; % [left, bottom, width, height]
subplot(3,2,[1,2,3,4]);
% axes('Position', posLarge);
% axis off;
t = title(sprintf('3D Octopus Matching with $\\rho=%.1f, \\sum P_{ij}=%.3f$', 1.5, 0.832), 'Interpreter', 'latex');

% Set the font size
set(t, 'FontSize', 20); % You can adjust the size (14 in this case) as needed


filename = 'highlight_indices_1.5.json';
%filename = 'highlight_indices_sgwnew_1.5.json';
fid = fopen(filename); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
data = jsondecode(str);
    
% Access your data
sub_highlightIndices1 = data.sub_highlightIndices1+1;
sub_highlightIndices2 = data.sub_highlightIndices2+1;
sub_highlightIndices1_body = data.sub_highlightIndices1_body+1;
sub_highlightIndices2_body = data.sub_highlightIndices2_body+1;
sub_highlightIndices1_leg1 = data.sub_highlightIndices1_leg1+1;
sub_highlightIndices2_leg1 = data.sub_highlightIndices2_leg1+1;
sub_highlightIndices1_leg2 = data.sub_highlightIndices1_leg2+1;
sub_highlightIndices2_leg2 = data.sub_highlightIndices2_leg2+1;
sub_highlightIndices1_leg3 = data.sub_highlightIndices1_leg3+1;
sub_highlightIndices2_leg3 = data.sub_highlightIndices2_leg3+1;
sub_highlightIndices1_leg4 = data.sub_highlightIndices1_leg4+1;
sub_highlightIndices2_leg4 = data.sub_highlightIndices2_leg4+1;
sub_highlightIndices1_leg5 = data.sub_highlightIndices1_leg5+1;
sub_highlightIndices2_leg5 = data.sub_highlightIndices2_leg5+1;
sub_highlightIndices1_leg6 = data.sub_highlightIndices1_leg6+1;
sub_highlightIndices2_leg6 = data.sub_highlightIndices2_leg6+1;
sub_highlightIndices1_leg7 = data.sub_highlightIndices1_leg7+1;
sub_highlightIndices2_leg7 = data.sub_highlightIndices2_leg7+1;
sub_highlightIndices1_leg8 = data.sub_highlightIndices1_leg8+1;
sub_highlightIndices2_leg8 = data.sub_highlightIndices2_leg8+1;

% Define the highlight indices for each set
highlightIndicesSets = {
    {sub_highlightIndices1_body, sub_highlightIndices2_body},
    {sub_highlightIndices1_leg1, sub_highlightIndices2_leg1},
    {sub_highlightIndices1_leg2, sub_highlightIndices2_leg2},
    {sub_highlightIndices1_leg3, sub_highlightIndices2_leg3},
    {sub_highlightIndices1_leg4, sub_highlightIndices2_leg4},
    {sub_highlightIndices1_leg5, sub_highlightIndices2_leg5},
    {sub_highlightIndices1_leg6, sub_highlightIndices2_leg6},
    {sub_highlightIndices1_leg7, sub_highlightIndices2_leg7},
    {sub_highlightIndices1_leg8, sub_highlightIndices2_leg8}
};
    
% Define the highlight colors for each set
highlightColorsSets = {
    {'red','red'},
    {[0,100,0]/255,[0,100,0]/255},
    {'blue', 'blue'},
    {'magenta', 'magenta'},
    {[255,215,0]/255, [255,215,0]/255},
    {[0,139,139]/255, [0,139,139]/255},
    {'black', 'black'},
    {[0.5, 0, 0.5], [0.5, 0, 0.5]},% purple
    {[255,140,0]/255,[255,140,0]/255} % orange
};


% Loop over each set of highlight indices and colors
for i = 1:length(highlightIndicesSets)
    highlightIndices = highlightIndicesSets{i};
    highlightColors = highlightColorsSets{i};

    % Call the plot function for each set
    [handles, allVertices] = plotMultipleOFFWithHighlights(filenames, fileColors, highlightIndices, highlightColors);
    
    % Depending on how plotMultipleOFFWithHighlights works, you might need to hold on
    hold on;
end


% Next, create the smaller subplot for the 0.5 case
subplot(3, 2, 5); % Bottom left position
t = title(sprintf('$\\rho=%.1f, \\sum P_{ij}=%.3f$', 0.2, 0.491), 'Interpreter', 'latex');
set(gca, 'Position', [0.23, 0.13, 0.3, 0.3]);


% Set the font size
set(t, 'FontSize', 20); % You can adjust the size (14 in this case) as needed


filename = 'highlight_indices_0.2.json';
fid = fopen(filename); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
data = jsondecode(str);
    
% Access your data
sub_highlightIndices1 = data.sub_highlightIndices1+1;
sub_highlightIndices2 = data.sub_highlightIndices2+1;
sub_highlightIndices1_body = data.sub_highlightIndices1_body+1;
sub_highlightIndices2_body = data.sub_highlightIndices2_body+1;
sub_highlightIndices1_leg1 = data.sub_highlightIndices1_leg1+1;
sub_highlightIndices2_leg1 = data.sub_highlightIndices2_leg1+1;
sub_highlightIndices1_leg2 = data.sub_highlightIndices1_leg2+1;
sub_highlightIndices2_leg2 = data.sub_highlightIndices2_leg2+1;
sub_highlightIndices1_leg3 = data.sub_highlightIndices1_leg3+1;
sub_highlightIndices2_leg3 = data.sub_highlightIndices2_leg3+1;
sub_highlightIndices1_leg4 = data.sub_highlightIndices1_leg4+1;
sub_highlightIndices2_leg4 = data.sub_highlightIndices2_leg4+1;
sub_highlightIndices1_leg5 = data.sub_highlightIndices1_leg5+1;
sub_highlightIndices2_leg5 = data.sub_highlightIndices2_leg5+1;
sub_highlightIndices1_leg6 = data.sub_highlightIndices1_leg6+1;
sub_highlightIndices2_leg6 = data.sub_highlightIndices2_leg6+1;
sub_highlightIndices1_leg7 = data.sub_highlightIndices1_leg7+1;
sub_highlightIndices2_leg7 = data.sub_highlightIndices2_leg7+1;
sub_highlightIndices1_leg8 = data.sub_highlightIndices1_leg8+1;
sub_highlightIndices2_leg8 = data.sub_highlightIndices2_leg8+1;

% Define the highlight indices for each set
highlightIndicesSets = {
    {sub_highlightIndices1_body, sub_highlightIndices2_body},
    {sub_highlightIndices1_leg1, sub_highlightIndices2_leg1},
    {sub_highlightIndices1_leg2, sub_highlightIndices2_leg2},
    {sub_highlightIndices1_leg3, sub_highlightIndices2_leg3},
    {sub_highlightIndices1_leg4, sub_highlightIndices2_leg4},
    {sub_highlightIndices1_leg5, sub_highlightIndices2_leg5},
    {sub_highlightIndices1_leg6, sub_highlightIndices2_leg6},
    {sub_highlightIndices1_leg7, sub_highlightIndices2_leg7},
    {sub_highlightIndices1_leg8, sub_highlightIndices2_leg8}
};
    
% Define the highlight colors for each set
highlightColorsSets = {
    {'red','red'},
    {[0,100,0]/255,[0,100,0]/255},
    {'blue', 'blue'},
    {'magenta', 'magenta'},
    {[255,215,0]/255, [255,215,0]/255},
    {[0,139,139]/255, [0,139,139]/255},
    {'black', 'black'},
    {[0.5, 0, 0.5], [0.5, 0, 0.5]},% purple
    {[255,140,0]/255,[255,140,0]/255} % orange
};


% Loop over each set of highlight indices and colors
for i = 1:length(highlightIndicesSets)
    highlightIndices = highlightIndicesSets{i};
    highlightColors = highlightColorsSets{i};

    % Call the plot function for each set
    [handles, allVertices] = plotMultipleOFFWithHighlights(filenames, fileColors, highlightIndices, highlightColors);
    
    % Depending on how plotMultipleOFFWithHighlights works, you might need to hold on
    hold on;
end

% Finally, create the smaller subplot for the 2.0 case
subplot(3, 2, 6); % Bottom right position
% Create the title
t = title(sprintf('$\\rho=%.1f, \\sum P_{ij}=%.3f$', 2.0, 0.984), 'Interpreter', 'latex');
set(gca, 'Position', [0.53, 0.13, 0.3, 0.3]);

% Set the font size
set(t, 'FontSize', 20); % You can adjust the size (14 in this case) as needed

filename = 'highlight_indices_2.0.json';
fid = fopen(filename); 
raw = fread(fid, inf); 
str = char(raw'); 
fclose(fid); 
data = jsondecode(str);
    
% Access your data
sub_highlightIndices1 = data.sub_highlightIndices1+1;
sub_highlightIndices2 = data.sub_highlightIndices2+1;
sub_highlightIndices1_body = data.sub_highlightIndices1_body+1;
sub_highlightIndices2_body = data.sub_highlightIndices2_body+1;
sub_highlightIndices1_leg1 = data.sub_highlightIndices1_leg1+1;
sub_highlightIndices2_leg1 = data.sub_highlightIndices2_leg1+1;
sub_highlightIndices1_leg2 = data.sub_highlightIndices1_leg2+1;
sub_highlightIndices2_leg2 = data.sub_highlightIndices2_leg2+1;
sub_highlightIndices1_leg3 = data.sub_highlightIndices1_leg3+1;
sub_highlightIndices2_leg3 = data.sub_highlightIndices2_leg3+1;
sub_highlightIndices1_leg4 = data.sub_highlightIndices1_leg4+1;
sub_highlightIndices2_leg4 = data.sub_highlightIndices2_leg4+1;
sub_highlightIndices1_leg5 = data.sub_highlightIndices1_leg5+1;
sub_highlightIndices2_leg5 = data.sub_highlightIndices2_leg5+1;
sub_highlightIndices1_leg6 = data.sub_highlightIndices1_leg6+1;
sub_highlightIndices2_leg6 = data.sub_highlightIndices2_leg6+1;
sub_highlightIndices1_leg7 = data.sub_highlightIndices1_leg7+1;
sub_highlightIndices2_leg7 = data.sub_highlightIndices2_leg7+1;
sub_highlightIndices1_leg8 = data.sub_highlightIndices1_leg8+1;
sub_highlightIndices2_leg8 = data.sub_highlightIndices2_leg8+1;

% Define the highlight indices for each set
highlightIndicesSets = {
    {sub_highlightIndices1_body, sub_highlightIndices2_body},
    {sub_highlightIndices1_leg1, sub_highlightIndices2_leg1},
    {sub_highlightIndices1_leg2, sub_highlightIndices2_leg2},
    {sub_highlightIndices1_leg3, sub_highlightIndices2_leg3},
    {sub_highlightIndices1_leg4, sub_highlightIndices2_leg4},
    {sub_highlightIndices1_leg5, sub_highlightIndices2_leg5},
    {sub_highlightIndices1_leg6, sub_highlightIndices2_leg6},
    {sub_highlightIndices1_leg7, sub_highlightIndices2_leg7},
    {sub_highlightIndices1_leg8, sub_highlightIndices2_leg8}
};
    
% Define the highlight colors for each set
highlightColorsSets = {
    {'red','red'},
    {[0,100,0]/255,[0,100,0]/255},
    {'blue', 'blue'},
    {'magenta', 'magenta'},
    {[255,215,0]/255, [255,215,0]/255},
    {[0,139,139]/255, [0,139,139]/255},
    {'black', 'black'},
    {[0.5, 0, 0.5], [0.5, 0, 0.5]},% purple
    {[255,140,0]/255,[255,140,0]/255} % orange
};


% Loop over each set of highlight indices and colors
for i = 1:length(highlightIndicesSets)
    highlightIndices = highlightIndicesSets{i};
    highlightColors = highlightColorsSets{i};

    % Call the plot function for each set
    [handles, allVertices] = plotMultipleOFFWithHighlights(filenames, fileColors, highlightIndices, highlightColors);
    
    % Depending on how plotMultipleOFFWithHighlights works, you might need to hold on
    hold on;
end

hold off;


