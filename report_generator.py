# report_generator.py
import matplotlib
matplotlib.use('Agg')  # Add this line BEFORE importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import numpy as np
import pickle
import os
import matplotlib.image as mpimg

def calculate_a3p(df, pitcher_name):
    df_pitcher = df[df['Pitcher'] == pitcher_name]
    df_pitcher_three_pitches = df_pitcher[df_pitcher['PitchofPA'] >= 3]
    if df_pitcher_three_pitches.empty:
        return 0
    a3p = (
        df_pitcher_three_pitches[
            (df_pitcher_three_pitches['Strikes'] >= 2) |
            (df_pitcher_three_pitches['PitchCall'] == 'InPlay')
        ].shape[0] / df_pitcher_three_pitches.shape[0]
    ) * 100
    return a3p

def load_models(model_dir):
    xgb_models = {}
    rf_models = {}
    types = [
        ('Fastball', 'fb'), ('Sinker', 'fb'), ('TwoSeamFastball', 'fb'), ('FourSeamFastball', 'fb'),
        ('Curveball', 'cb'), ('Slider', 'sl'), ('Cutter', 'sl'),
        ('ChangeUp', 'ch'), ('Splitter', 'ch')
    ]
    for pitch_type, suffix in types:
        xgb_path = os.path.join(model_dir, f"xgb_model{suffix}.sav")
        rf_path = os.path.join(model_dir, f"rfc_model{suffix}.sav")
        xgb_models[pitch_type] = pickle.load(open(xgb_path, 'rb'))
        rf_models[pitch_type] = pickle.load(open(rf_path, 'rb'))
    return xgb_models, rf_models

def generate_report(df, pitcher_name, innings_pitched):
    if 'TaggedPitchType' not in df.columns:
        raise KeyError("Column 'TaggedPitchType' not found in DataFrame")

    output_path = os.path.join("outputs", f"{pitcher_name.replace(',', '').replace(' ', '_')}_report.pdf")
    model_dir = "models"
    logo_path = os.path.join("static", "Wareham.jpg")
    
    report_colors = {
        "background": "#f4f4f9",
        "table_bg": "#ffffff",
        "title_color": "#003366",
        "text_color": "#333333",
        "accent1": "#007acc",
        "accent2": "#d9534f",
        "accent3": "#5cb85c",
        "strike_zone_border": "#000000"
    }


    pitcher_data = df[df['Pitcher'] == pitcher_name]
    pitcher_data = pitcher_data[pitcher_data['TaggedPitchType'].notna() & (pitcher_data['TaggedPitchType'] != 'Undefined')]

    xgb_models, rf_models = load_models(model_dir)

    pitch_type_counts = pitcher_data['TaggedPitchType'].value_counts()
    pitch_type_averages = pitcher_data.groupby('TaggedPitchType')[['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'RelHeight', 'RelSide', 'Extension']].mean()
    pitch_type_averages.reset_index(inplace=True)
    pitch_type_averages['differential_break'] = abs(pitch_type_averages['InducedVertBreak'].abs() - pitch_type_averages['HorzBreak'].abs())
    pitch_type_averages['ABS_Horizontal'] = pitch_type_averages['HorzBreak'].abs()
    pitch_type_averages['ABS_RelSide'] = pitch_type_averages['RelSide'].abs()

    a3p = calculate_a3p(df, pitcher_name)
    strikes = pitcher_data[pitcher_data['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'FoulBall', 'InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])]
    strike_percentages = strikes.groupby('TaggedPitchType')['PitchCall'].count() / pitch_type_counts * 100
    strike_percentages = strike_percentages.fillna(0)

    pitch_type_max_velocity = pitcher_data.groupby('TaggedPitchType')['RelSpeed'].max().reset_index()
    pitch_type_max_velocity.rename(columns={'RelSpeed': 'MaxVelo'}, inplace=True)
    pitch_type_averages = pitch_type_averages.merge(pitch_type_max_velocity, on='TaggedPitchType', how='left')

    cols = pitch_type_averages.columns.tolist()
    cols.insert(cols.index('RelSpeed') + 1, cols.pop(cols.index('MaxVelo')))
    pitch_type_averages = pitch_type_averages[cols]

    hits = len(pitcher_data[pitcher_data['PlayResult'].isin(['Single', 'Double', 'Triple', 'HomeRun'])])
    total_pitches = len(pitcher_data)
    strikeouts = len(pitcher_data[pitcher_data['KorBB'] == 'Strikeout'])
    walks = len(pitcher_data[pitcher_data['KorBB'] == 'Walk'])

    outs_pitch = (pitcher_data['Outs'].diff() > 0).sum()
    outs_play = len(pitcher_data[pitcher_data['PlayResult'].isin(['Out', 'Sacrifice'])])
    total_outs = outs_pitch + outs_play

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True))

    total_swings = pitcher_data[pitcher_data['PitchCall'].isin(['StrikeSwinging', 'FoulBall', 'InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])].groupby('TaggedPitchType').size()
    whiffs = pitcher_data[pitcher_data['PitchCall'] == 'StrikeSwinging'].groupby('TaggedPitchType').size()
    whiff_percentages = (whiffs / total_swings) * 100
    whiff_percentages = whiff_percentages.fillna(0)

    whiff_constants = {
        'Fastball': 0.1570507714710533618927,
        'Sinker': 0.160227714710533618927,
        'TwoSeamFastball': 0.17714710533618927,
        'FourSeamFastball': 0.17714710533618927,
        'Curveball': 0.203832322178011536598206,
        'Slider': 0.2822236396014690399,
        'Cutter': 0.32232236396014690399,
        'ChangeUp': 0.28232322178011536598206,
        'Splitter': 0.2878011536598206
    }

    df_stuff = pd.DataFrame()
    for pitch in pitch_type_averages['TaggedPitchType'].unique():
        if pitch in whiff_constants:
            features = ['RelSpeed', 'SpinRate', 'differential_break', 'RelHeight', 'ABS_RelSide', 'Extension'] if pitch in ['Fastball', 'Sinker', 'TwoSeamFastball', 'FourSeamFastball'] else ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'ABS_Horizontal', 'RelHeight', 'ABS_RelSide', 'Extension']
            pitch_metrics = pitch_type_averages[pitch_type_averages['TaggedPitchType'] == pitch][features].copy()
            model_1 = xgb_models[pitch]
            model_2 = rf_models[pitch]
            proba_predictions_1 = model_1.predict_proba(pitch_metrics)[:, 1]
            proba_predictions_2 = model_2.predict_proba(pitch_metrics)[:, 1]
            proba_predictions = (proba_predictions_1 + proba_predictions_2) / 2
            df_stuff[pitch] = (proba_predictions / whiff_constants[pitch]) * 100

    df_stuff = df_stuff.round(2)
    df_stuff.rename(columns={
        'Fastball': 'Fastball Stuff+',
        'FourSeamFastball': 'Four-Seam Fastball Stuff+',
        'TwoSeamFastball': 'Two-Seam Stuff+',
        'Sinker': 'Sinker Stuff+',
        'Curveball': 'Curveball Stuff+',
        'Slider': 'Slider Stuff+',
        'Cutter': 'Cutter Stuff+',
        'ChangeUp': 'Changeup Stuff+',
        'Splitter': 'Splitter Stuff+'
    }, inplace=True)

    pitch_type_averages_1 = pitch_type_averages[['TaggedPitchType', 'RelSpeed','MaxVelo','SpinRate', 'InducedVertBreak', 'HorzBreak', 'RelHeight', 'RelSide', 'Extension']]
    pitch_type_averages_1.rename(columns={'TaggedPitchType': 'Pitch Type', 'RelSpeed': 'Avg Velo'}, inplace=True)

    logo_img = mpimg.imread(logo_path)

    with PdfPages(output_path) as pdf:
        fig = plt.figure(figsize=(11, 17))
        fig.patch.set_facecolor(report_colors["background"])
        gs = GridSpec(8, 8, figure=fig)

        ax1 = fig.add_subplot(gs[0:2, 0:4])
        ax1.pie(pitch_type_counts, labels=pitch_type_counts.index, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Pitch Types Distribution', fontsize=16)
        ax1.set_position([0.175, 0.78, 0.35, 0.15])

        ax2 = fig.add_subplot(gs[0:2, 4:8])
        ax2.axis('off')
        basic_stats_data = [
            ['Total Pitches', total_pitches],
            ['Innings Pitched', innings_pitched],
            ['Strikeouts', strikeouts],
            ['Walks', walks],
            ['Hits', hits],
            ['Strike %', round(len(strikes) / total_pitches * 100, 2)],
            ['First Pitch Strike %', round(len(pitcher_data[pitcher_data['PitchofPA'] == 1][pitcher_data['PitchCall'].isin(['StrikeCalled', 'StrikeSwinging', 'FoulBall', 'InPlay', 'FoulBallNotFieldable', 'FoulBallFieldable'])]) / len(pitcher_data[pitcher_data['PitchofPA'] == 1]) * 100, 2)],
            ['Ahead After 3 Pitches %', round(a3p, 2)]
        ]
        basic_stats_table = ax2.table(cellText=basic_stats_data, cellLoc='left', loc='center', colLabels=None, bbox=[0, 0, 1, 1])
        basic_stats_table.auto_set_font_size(False)
        basic_stats_table.set_fontsize(12)
        basic_stats_table.auto_set_column_width(col=list(range(2)))
        ax2.set_title("Outing Summary", fontsize=16, color=report_colors["title_color"])
        ax2.set_position([0.66, 0.54, .35, 0.17])

        ax3 = fig.add_subplot(gs[2:4, 0:8])
        ax3.axis('off')



        pitch_type_averages_table = ax3.table(cellText=pitch_type_averages_1.round(1).values,
                                              colLabels=pitch_type_averages_1.columns,
                                              cellLoc='center', loc='center')
        pitch_type_averages_table.auto_set_font_size(False)
        pitch_type_averages_table.set_fontsize(12)
        pitch_type_averages_table.auto_set_column_width(col=list(range(len(pitch_type_averages_1.columns))))
        pitch_type_averages_table.scale(1.2, 2.2)
        pitch_type_averages_table.set_fontsize(13)
        ax3.set_position([0.42, 0.25, 0.35, 0.12])
        
        unique_pitches = pitcher_data['TaggedPitchType'].unique()
        pitch_colors = dict(zip(unique_pitches, plt.cm.get_cmap('tab10')(range(len(unique_pitches)))))
        
        ax4 = fig.add_subplot(gs[4:6, 0:5])
        strike_zone = Rectangle((-0.83, 1.5), 1.66, 2.1, linewidth=1, edgecolor='k', facecolor='none')
        ax4.add_patch(strike_zone)
        ax4.set_xlim(-2, 2)
        ax4.set_ylim(0, 4.5)
        ax4.set_xlabel("Plate Side (in feet)", fontsize=10)
        ax4.set_ylabel("Plate Height (in feet)", fontsize=10)

        ax4.set_aspect('equal')
        # Match pitch colors with ax8
        for pitch in unique_pitches:
            filtered_data = pitcher_data[pitcher_data['TaggedPitchType'] == pitch]
            ax4.scatter(
                filtered_data['PlateLocSide'],
                filtered_data['PlateLocHeight'],
                s=80,
                color=pitch_colors[pitch],
                alpha=0.6,
                label=pitch
                )

        ax4.legend(title="Pitch Types", fontsize=10)
        ax4.set_title("Pitch Locations (pitcher's perspective)", fontsize=16)
        ax4.set_position([0.12, 0.47, 0.5, 0.3])


        ax5 = fig.add_subplot(gs[4:6, 5:8])
        ax5.axis('off')
        strike_percentages_data = [[pitch_type, round(strike_percentages[pitch_type], 2)] for pitch_type in strike_percentages.index]
        strike_percentages_table = ax5.table(cellText=strike_percentages_data,
                                             colLabels=['Pitch Type', 'Strike %'],
                                             cellLoc='center', loc='center')
        strike_percentages_table.auto_set_font_size(False)
        strike_percentages_table.set_fontsize(12)
        strike_percentages_table.auto_set_column_width(col=list(range(2)))
        strike_percentages_table.scale(1.2, 1.8)
        strike_percentages_table.set_fontsize(13)
        ax5.set_position([0.76, 0.43, 0.35, 0.15])
        #0.72
        ax6 = fig.add_subplot(gs[6:8, 0:4])
        ax6.axis('off')
        whiff_percentages_data = [[pitch_type, round(whiff_percentages[pitch_type], 2)] for pitch_type in whiff_percentages.index]
        whiff_percentages_table = ax6.table(cellText=whiff_percentages_data,
                                            colLabels=['Pitch Type', 'Whiff %'],
                                            cellLoc='center', loc='center')
        whiff_percentages_table.auto_set_font_size(False)
        whiff_percentages_table.set_fontsize(12)
        whiff_percentages_table.auto_set_column_width(col=list(range(2)))
        whiff_percentages_table.scale(1.2, 1.8)
        whiff_percentages_table.set_fontsize(13)
        #0.52
        ax6.set_position([0.56, 0.43, 0.35, 0.15])

        '''
        #This is the old Stuff+ table
        ax7 = fig.add_subplot(gs[6:8, 4:8])
        ax7.axis('off')
        stuff_metrics_table = ax7.table(cellText=df_stuff.round(1).values,
                                        colLabels=df_stuff.columns,
                                        cellLoc='center', loc='center')
        stuff_metrics_table.auto_set_font_size(False)
        stuff_metrics_table.set_fontsize(12)
        stuff_metrics_table.auto_set_column_width(col=list(range(len(df_stuff.columns))))
        stuff_metrics_table.scale(1.2, 1.6)
        stuff_metrics_table.set_fontsize(13)
        #update figure position
        ax7.set_position([0.20, 0.36, 0.8, 0.15])
        '''
        # Clean horizontal Stuff+ scale visualization (streamlined)
        ax7 = fig.add_subplot(gs[6:8, 4:8])
        ax7.set_facecolor('none')  # Transparent background
        ax7.set_frame_on(False)
        ax7.set_yticks([])
        ax7.set_xticks([])
        ax7.set_title("Stuff+ Benchmark (100 = NCAA Avg)", fontsize=13, color=report_colors["title_color"])


        mean = 100
        std = 10
        ax7.axhline(y=0, color='black', linewidth=2)

        # Central tick for mean and its label
        ax7.plot(mean, 0, marker='|', color='black', markersize=22, zorder=2)
        ax7.text(mean, -0.2, "100", ha='center', va='top', fontsize=9, color='black')

        # ±1σ to ±3σ ticks and their labels
        for i in range(1, 4):
            for offset in [-1, 1]:
                val = mean + offset * i * std
                ax7.plot(val, 0, marker='|', color='gray', markersize=12, zorder=1)
                ax7.text(val, -0.2, f"{val}", ha='center', va='top', fontsize=8, color='gray')

        # Colored Stuff+ markers and labels (just values, horizontally)
        stuffplus_colors = dict(zip(df_stuff.columns, plt.cm.get_cmap('tab10')(range(len(df_stuff.columns)))))
        for pitch, score in df_stuff.iloc[0].items():
            ax7.plot(score, 0, marker='|', color=stuffplus_colors[pitch], markersize=26, linewidth=4, zorder=3)
            ax7.text(score, 0.2, f"{score:.1f}", fontsize=9, ha='center', va='bottom', color=stuffplus_colors[pitch])

        ax7.set_xlim(mean - 3*std - 5, mean + 3*std + 5)
        ax7.set_ylim(-0.4, 0.4)
        ax7.set_position([0.20, 0.36, 0.8, 0.05])  # Controls width & height

        ax8 = fig.add_subplot(gs[2:4, 0:3])
        ax8.set_facecolor(report_colors["table_bg"])
        ax8.set_title("Pitch Movement Plot", fontsize=16, color=report_colors["title_color"])
        ax8.set_xlabel("Horizontal Break (inches)", fontsize=6, color=report_colors["text_color"])
        ax8.set_ylabel("Induced Vertical Break (inches)", fontsize=8, color=report_colors["text_color"])
        ax8.tick_params(colors=report_colors["text_color"])

        unique_pitches = pitcher_data['TaggedPitchType'].unique()
        pitch_colors = dict(zip(unique_pitches, plt.cm.get_cmap('tab10')(range(len(unique_pitches)))))

        for pitch in unique_pitches:
            subset = pitcher_data[pitcher_data['TaggedPitchType'] == pitch]
            ax8.scatter(
                subset['HorzBreak'],
                subset['InducedVertBreak'],
                label=pitch,
                alpha=0.6,
                s=60,
                color=pitch_colors[pitch]
            )

        ax8.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax8.axvline(0, color='gray', linestyle='--', linewidth=0.5)
        #ax8.legend(loc="lower left", fontsize=9, title="Pitch Type")
        ax8.set_xlim(-25, 25)
        ax8.set_ylim(-25, 25)
        ax8.set_position([0.68, 0.76, 0.30, 0.16])
        

        #ax_logo = fig.add_axes([0.61, 0.72, 0.28, 0.28])
        #ax_logo.axis('off')
        #ax_logo.imshow(logo_img)

        plt.suptitle(f"{pitcher_name} Live 04/24", fontsize=30, x = 0.65, y= 0.98)
        pdf.savefig(fig, bbox_inches='tight')
        #pdf.close()

    return output_path
