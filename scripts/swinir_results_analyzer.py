#!/usr/bin/env python3
"""
Sistema de Análisis de Resultados SwinIR
Genera gráficos de violín y tablas de resumen para artículo científico
Adaptado específicamente para el proyecto SwinIR con datos de evaluation_results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import glob
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class SwinIRResultsAnalyzer:
    """Analizador de resultados SwinIR para artículo científico"""
    
    def __init__(self, evaluation_dir="evaluation_results"):
        """
        Inicializa el analizador
        
        Args:
            evaluation_dir: Directorio con todos los resultados de evaluación de SwinIR
        """
        self.evaluation_dir = Path(evaluation_dir)
        self.metrics = ['psnr', 'ssim', 'ms_ssim', 'mse', 'perceptual_index']
        
        # Configuración de modelos SwinIR y sus parámetros
        self.model_configs = {
            '64to128': {
                'scale': 2, 
                'input_size': 64,
                'output_size': 128,
                'params': {
                    'embed_dim': 180,
                    'depths': [6, 6, 6, 6, 6, 6],
                    'num_heads': [6, 6, 6, 6, 6, 6],
                    'window_size': 8,
                    'mlp_ratio': 2,
                    'resi_connection': '1conv'
                }
            },
            '128to256': {
                'scale': 2,
                'input_size': 128, 
                'output_size': 256,
                'params': {
                    'embed_dim': 180,
                    'depths': [6, 6, 6, 6, 6, 6],
                    'num_heads': [6, 6, 6, 6, 6, 6],
                    'window_size': 8,
                    'mlp_ratio': 2,
                    'resi_connection': '1conv'
                }
            },
            '256to512': {
                'scale': 2,
                'input_size': 256,
                'output_size': 512,
                'params': {
                    'embed_dim': 180,
                    'depths': [6, 6, 6, 6],
                    'num_heads': [6, 6, 6, 6],
                    'window_size': 8,
                    'mlp_ratio': 2,
                    'resi_connection': '1conv'
                }
            },
            '512to1024': {
                'scale': 2,
                'input_size': 512,
                'output_size': 1024,
                'params': {
                    'embed_dim': 60,
                    'depths': [4, 4, 4],
                    'num_heads': [4, 4, 4],
                    'window_size': 8,
                    'mlp_ratio': 2,
                    'resi_connection': '1conv'
                }
            }
        }
        
    def load_all_metrics(self):
        """Carga todas las métricas de todos los modelos SwinIR"""
        all_data = []
        
        # Buscar todos los directorios de modelos
        for model_dir in self.evaluation_dir.glob("*"):
            if not model_dir.is_dir() or model_dir.name == 'logs':
                continue
                
            model_name = model_dir.name
            base_model = model_name
            
            # Buscar archivos CSV de métricas en Validation_Images
            validation_dir = model_dir / "Validation_Images"
            if not validation_dir.exists():
                continue
                
            for csv_file in validation_dir.glob("*_metrics_*.csv"):
                # Determinar tipo (color/grayscale)
                if "color" in csv_file.name:
                    image_type = "color"
                elif "grayscale" in csv_file.name:
                    image_type = "grayscale"
                else:
                    continue
                
                try:
                    df = pd.read_csv(csv_file)
                    
                    # Agregar información del modelo
                    df['model_name'] = model_name
                    df['base_model'] = base_model
                    df['image_type'] = image_type
                    df['scale_factor'] = self.model_configs.get(base_model, {}).get('scale', 1)
                    df['target_resolution'] = self.model_configs.get(base_model, {}).get('output_size', 0)
                    df['input_resolution'] = self.model_configs.get(base_model, {}).get('input_size', 0)
                    
                    all_data.append(df)
                    print(f"✅ Cargado: {model_name} ({image_type}) - {len(df)} imágenes")
                    
                except Exception as e:
                    print(f"❌ Error cargando {csv_file}: {e}")
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"\n📊 Total de datos cargados: {len(combined_df)} registros")
            return combined_df
        else:
            print("❌ No se encontraron datos de métricas")
            return None
    
    def load_all_timing(self):
        """Carga todos los datos de timing de SwinIR"""
        all_timing = []
        
        # Buscar archivos de timing en realistic_timing_results
        for model_dir in self.evaluation_dir.glob("*"):
            if not model_dir.is_dir() or model_dir.name == 'logs':
                continue
                
            timing_dir = model_dir / "realistic_timing_results"
            if not timing_dir.exists():
                continue
                
            model_name = model_dir.name
            base_model = model_name
            
            # Buscar archivos de timing CSV
            for timing_file in timing_dir.glob("*_timing.csv"):
                try:
                    df = pd.read_csv(timing_file)
                    
                    # Extraer información del nombre del archivo
                    filename = timing_file.stem
                    
                    # Determinar dispositivo (GPU/CPU)
                    if '_GPU_' in filename:
                        device = 'gpu'
                    elif '_CPU_' in filename:
                        device = 'cpu'
                    else:
                        device = 'unknown'
                    
                    df['model_name'] = model_name
                    df['base_model'] = base_model
                    df['device'] = device
                    df['scale_factor'] = self.model_configs.get(base_model, {}).get('scale', 1)
                    df['input_size'] = self.model_configs.get(base_model, {}).get('input_size', 0)
                    df['output_size'] = self.model_configs.get(base_model, {}).get('output_size', 0)
                    
                    all_timing.append(df)
                    print(f"✅ Timing cargado: {model_name} ({device})")
                    
                except Exception as e:
                    print(f"❌ Error cargando timing {timing_file}: {e}")
        
        if all_timing:
            combined_timing = pd.concat(all_timing, ignore_index=True)
            print(f"\n⏱️  Total timing data: {len(combined_timing)} registros")
            return combined_timing
        else:
            print("❌ No se encontraron datos de timing")
            return None
    
    def create_violin_plot_all_models(self, metrics_df, output_path="swinir_violin_plot_all_models.png"):
        """
        Crea gráfico de violín para todos los modelos SwinIR
        
        Args:
            metrics_df: DataFrame con todas las métricas
            output_path: Ruta para guardar el gráfico
        """
        # Configurar fuente Computer Modern y tamaños más grandes
        plt.rcParams.update({
            'font.family': ['serif'],
            'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
            'font.size': 16,           # Tamaño base aumentado
            'axes.titlesize': 18,      # Título principal
            'axes.labelsize': 16,      # Etiquetas de ejes
            'xtick.labelsize': 16,     # Etiquetas del eje X
            'ytick.labelsize': 16,     # Etiquetas del eje Y
            'legend.fontsize': 16,     # Leyenda
            'figure.titlesize': 22     # Título de figura
        })
        
        # Filtrar solo datos de color para consistencia
        df_color = metrics_df[metrics_df['image_type'] == 'color'].copy()
        
        if df_color.empty:
            print("❌ No se encontraron datos de color para los modelos")
            return
        
        # Obtener todos los modelos únicos
        all_models = sorted(df_color['base_model'].unique())
        
        # Configurar estilo similar al original ESRGAN
        sns.set_style("whitegrid")
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Nombres de métricas en español
        metric_names = {
            'psnr': 'PSNR (dB)',
            'ssim': 'SSIM',
            'ms_ssim': 'MS-SSIM',
            'mse': 'Error Cuadrático Medio'
        }
        
        # 1. Crear gráfico principal con 4 métricas (2x2)
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.ravel()
        
        # Crear violín para las 4 métricas principales
        metrics_to_plot = ['psnr', 'ssim', 'ms_ssim', 'mse']
        
        for i, metric in enumerate(metrics_to_plot):
            if metric not in df_color.columns:
                continue
            
            ax = axes[i]
            
            # Crear el violin plot
            violin_parts = ax.violinplot(
                [df_color[df_color['base_model'] == model][metric].values 
                 for model in all_models],
                positions=range(len(all_models)),
                showmeans=True,
                showmedians=True,
                showextrema=True
            )
            
            # Personalizar colores
            for pc, color in zip(violin_parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Configurar ejes
            ax.set_xticks(range(len(all_models)))
            ax.set_xticklabels([f'{model}\n(×{self.model_configs[model]["scale"]})' 
                              for model in all_models], rotation=45)
            
            # Título y labels en español
            metric_display = metric_names[metric]
            if metric == 'mse':
                ax.set_ylabel(f'{metric_display} (menor es mejor)')
            else:
                ax.set_ylabel(f'{metric_display} (mayor es mejor)')
                
            ax.set_title(f'Distribución de {metric_display} por Factor de Escala', fontweight='bold', pad=20)
            ax.set_xlabel('Modelo (Factor de Escala)')
            ax.grid(True, alpha=0.3)
            
            # Agregar estadísticas en el gráfico
            stats_text = []
            for j, model in enumerate(all_models):
                model_data = df_color[df_color['base_model'] == model][metric]
                mean_val = model_data.mean()
                std_val = model_data.std()
                stats_text.append(f'{model}: μ={mean_val:.4f}±{std_val:.4f}')
            
            # Agregar texto de estadísticas
            stats_str = '\n'.join(stats_text)
            ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"🎻 Gráfico de violín principal guardado: {output_path}")
        
        # 2. Crear gráfico separado para índice perceptual
        perceptual_path = output_path.replace('.png', '_perceptual.png')
        
        if 'perceptual_index' in df_color.columns:
            fig, ax = plt.subplots(1, 1, figsize=(12, 8))
            
            # Crear el violin plot para índice perceptual
            violin_parts = ax.violinplot(
                [df_color[df_color['base_model'] == model]['perceptual_index'].values 
                 for model in all_models],
                positions=range(len(all_models)),
                showmeans=True,
                showmedians=True,
                showextrema=True
            )
            
            # Personalizar colores
            for pc, color in zip(violin_parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
            
            # Configurar ejes
            ax.set_xticks(range(len(all_models)))
            ax.set_xticklabels([f'{model}\n(×{self.model_configs[model]["scale"]})' 
                              for model in all_models], rotation=45)
            
            ax.set_ylabel('Índice Perceptual (menor es mejor)')
            ax.set_title('Distribución del Índice Perceptual por Factor de Escala', fontweight='bold', pad=20)
            ax.set_xlabel('Modelo (Factor de Escala)')
            ax.grid(True, alpha=0.3)
            
            # Agregar estadísticas
            stats_text = []
            for j, model in enumerate(all_models):
                model_data = df_color[df_color['base_model'] == model]['perceptual_index']
                mean_val = model_data.mean()
                std_val = model_data.std()
                stats_text.append(f'{model}: μ={mean_val:.4f}±{std_val:.4f}')
            
            stats_str = '\n'.join(stats_text)
            ax.text(0.02, 0.98, stats_str, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=12, 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(perceptual_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"🎻 Gráfico de índice perceptual guardado: {perceptual_path}")
        
        # Crear tabla estadística
        self._create_statistics_table_all_models(df_color, all_models, 
                                                output_path.replace('.png', '_stats.csv'))
        
        # Restaurar configuración por defecto
        plt.rcdefaults()
    
    def _create_statistics_table_all_models(self, df_color, models, output_path):
        """Crea tabla de estadísticas para todos los modelos"""
        stats_data = []
        
        for model in models:
            model_data = df_color[df_color['base_model'] == model]
            
            if len(model_data) == 0:
                continue
            
            row = {
                'Model': model,
                'Scale': f"×{self.model_configs[model]['scale']}",
                'Input_Size': f"{self.model_configs[model]['input_size']}px",
                'Output_Size': f"{self.model_configs[model]['output_size']}px",
                'Samples': len(model_data)
            }
            
            for metric in self.metrics:
                if metric in model_data.columns:
                    values = model_data[metric].dropna()
                    if len(values) > 0:
                        row[f'{metric}_mean'] = values.mean()
                        row[f'{metric}_std'] = values.std()
                        row[f'{metric}_median'] = values.median()
                        row[f'{metric}_min'] = values.min()
                        row[f'{metric}_max'] = values.max()
            
            stats_data.append(row)
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv(output_path, index=False)
        print(f"📊 Estadísticas guardadas: {output_path}")
    
    def create_summary_tables(self, metrics_df, timing_df, output_dir="swinir_summary_tables"):
        """
        Crea tablas de resumen para LaTeX
        
        Args:
            metrics_df: DataFrame con métricas
            timing_df: DataFrame con timing
            output_dir: Directorio para guardar tablas
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Tabla de métricas promedio por modelo (solo color)
        metrics_color = metrics_df[metrics_df['image_type'] == 'color']
        
        summary_metrics = []
        for model_name in metrics_color['model_name'].unique():
            model_data = metrics_color[metrics_color['model_name'] == model_name]
            base_model = model_data['base_model'].iloc[0]
            
            row = {
                'Model': model_name,
                'Scale': f"×{self.model_configs.get(base_model, {}).get('scale', '?')}",
                'Input_Res': f"{model_data['input_resolution'].iloc[0]}×{model_data['input_resolution'].iloc[0]}",
                'Target_Res': f"{model_data['target_resolution'].iloc[0]}×{model_data['target_resolution'].iloc[0]}",
                'Samples': len(model_data)
            }
            
            # Parámetros arquitectónicos SwinIR
            params = self.model_configs.get(base_model, {}).get('params', {})
            row['Embed_Dim'] = params.get('embed_dim', '?')
            row['Window_Size'] = params.get('window_size', '?')
            row['Depths'] = str(params.get('depths', '?'))
            row['Num_Heads'] = str(params.get('num_heads', '?'))
            row['MLP_Ratio'] = params.get('mlp_ratio', '?')
            row['Resi_Connection'] = params.get('resi_connection', '?')
            
            # Métricas
            for metric in self.metrics:
                if metric in model_data.columns:
                    values = model_data[metric].dropna()
                    if len(values) > 0:
                        row[f'{metric.upper()}_mean'] = f"{values.mean():.4f}"
                        row[f'{metric.upper()}_std'] = f"{values.std():.4f}"
            
            summary_metrics.append(row)
        
        metrics_summary_df = pd.DataFrame(summary_metrics)
        metrics_summary_df.to_csv(f"{output_dir}/metrics_summary.csv", index=False)
        
        # 2. Tabla de timing promedio
        if timing_df is not None:
            timing_summary = []
            
            for model_name in timing_df['model_name'].unique():
                model_timing = timing_df[timing_df['model_name'] == model_name]
                
                for device in ['gpu', 'cpu']:
                    device_data = model_timing[model_timing['device'] == device]
                    if len(device_data) == 0:
                        continue
                    
                    row = {
                        'Model': model_name,
                        'Device': device.upper(),
                        'Samples': len(device_data),
                        'Mean_Time_ms': f"{device_data['mean_time_ms'].mean():.2f}",
                        'Std_Time_ms': f"{device_data['mean_time_ms'].std():.2f}",
                        'FPS_mean': f"{device_data['fps'].mean():.2f}",
                        'Memory_MB': f"{device_data['memory_increase_mb'].mean():.1f}"
                    }
                    
                    # Agregar información específica de GPU si está disponible
                    if 'gpu_memory_increase_mb' in device_data.columns:
                        row['GPU_Memory_MB'] = f"{device_data['gpu_memory_increase_mb'].mean():.1f}"
                    
                    timing_summary.append(row)
            
            timing_summary_df = pd.DataFrame(timing_summary)
            timing_summary_df.to_csv(f"{output_dir}/timing_summary.csv", index=False)
        
        # 3. Tabla LaTeX-ready combinada
        self._create_latex_table_swinir(metrics_summary_df, timing_summary_df if timing_df is not None else None, 
                                       output_dir)
        
        print(f"📋 Tablas de resumen guardadas en: {output_dir}")
    
    def _create_latex_table_swinir(self, metrics_df, timing_df, output_dir):
        """Crea tabla formateada para LaTeX específica para SwinIR"""
        
        # Tabla de métricas principal
        latex_content = """
% Tabla de Métricas de Modelos SwinIR
\\begin{table*}[htbp]
\\centering
\\caption{Performance Metrics of SwinIR Models for Histopathology Super-Resolution}
\\label{tab:swinir_metrics}
\\begin{tabular}{|l|c|c|c|c|c|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Scale} & \\textbf{Resolution} & \\textbf{Architecture} & \\textbf{PSNR} & \\textbf{SSIM} & \\textbf{MS-SSIM} & \\textbf{Perceptual} & \\textbf{MSE} & \\textbf{Samples} \\\\
& & (Input→Target) & (ED, WS) & (dB) & & & Index & & \\\\
\\hline
"""
        
        for _, row in metrics_df.iterrows():
            model_name = row['Model'].replace('_', '\\_')
            resolution = f"{row['Input_Res']}→{row['Target_Res']}"
            architecture = f"({row['Embed_Dim']}, {row['Window_Size']})"
            
            latex_content += f"{model_name} & {row['Scale']} & {resolution} & {architecture} & "
            latex_content += f"{row.get('PSNR_mean', 'N/A')} & {row.get('SSIM_mean', 'N/A')} & "
            latex_content += f"{row.get('MS_SSIM_mean', 'N/A')} & {row.get('PERCEPTUAL_INDEX_mean', 'N/A')} & "
            latex_content += f"{row.get('MSE_mean', 'N/A')} & {row['Samples']} \\\\\n"
        
        latex_content += """\\hline
\\end{tabular}
\\begin{tablenotes}
\\item ED: Embedding Dimension, WS: Window Size
\\item All models use 6 transformer blocks with depth [6,6,6,6,6,6]
\\end{tablenotes}
\\end{table*}

"""
        
        # Tabla de timing si está disponible
        if timing_df is not None:
            latex_content += """
% Tabla de Timing SwinIR
\\begin{table}[htbp]
\\centering
\\caption{SwinIR Inference Time Comparison: GPU vs CPU}
\\label{tab:swinir_timing}
\\begin{tabular}{|l|c|c|c|c|c|}
\\hline
\\textbf{Model} & \\textbf{Device} & \\textbf{Time (ms)} & \\textbf{FPS} & \\textbf{RAM (MB)} & \\textbf{VRAM (MB)} \\\\
\\hline
"""
            
            for _, row in timing_df.iterrows():
                model_name = row['Model'].replace('_', '\\_')
                gpu_mem = row.get('GPU_Memory_MB', 'N/A')
                latex_content += f"{model_name} & {row['Device']} & {row['Mean_Time_ms']} & "
                latex_content += f"{row['FPS_mean']} & {row['Memory_MB']} & {gpu_mem} \\\\\n"
            
            latex_content += """\\hline
\\end{tabular}
\\end{table}
"""
        
        # Guardar archivo LaTeX
        with open(f"{output_dir}/swinir_tables_for_latex.tex", 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        print(f"📄 Tabla LaTeX guardada: {output_dir}/swinir_tables_for_latex.tex")
    
    def generate_comprehensive_report(self, output_dir="swinir_comprehensive_report"):
        """Genera reporte comprehensivo completo para SwinIR"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🚀 Generando reporte comprehensivo SwinIR...")
        
        # Cargar todos los datos
        print("\n📊 Cargando métricas...")
        metrics_df = self.load_all_metrics()
        
        print("\n⏱️  Cargando timing...")
        timing_df = self.load_all_timing()
        
        if metrics_df is None:
            print("❌ No se pudieron cargar las métricas")
            return
        
        # Crear gráfico de violín para todos los modelos
        print("\n🎻 Creando gráfico de violín...")
        self.create_violin_plot_all_models(
            metrics_df, 
            os.path.join(output_dir, "swinir_violin_plot_all_models.png")
        )
        
        # Crear tablas de resumen
        print("\n📋 Creando tablas de resumen...")
        self.create_summary_tables(metrics_df, timing_df, 
                                  os.path.join(output_dir, "summary_tables"))
        
        # Crear análisis estadístico adicional
        print("\n📈 Creando análisis estadístico...")
        self._create_additional_analysis(metrics_df, timing_df, output_dir)
        
        print(f"\n🎉 Reporte SwinIR completo generado en: {output_dir}")
    
    def _create_additional_analysis(self, metrics_df, timing_df, output_dir):
        """Crea análisis estadístico adicional para SwinIR"""
        
        # Configurar fuente Computer Modern
        plt.rcParams.update({
            'font.family': ['serif'],
            'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
            'font.size': 16,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16
        })
        
        # Análisis de correlaciones
        if metrics_df is not None:
            color_metrics = metrics_df[metrics_df['image_type'] == 'color']
            
            # Incluir resoluciones en el análisis de correlación
            correlation_columns = self.metrics + ['scale_factor', 'input_resolution', 'target_resolution']
            available_columns = [col for col in correlation_columns if col in color_metrics.columns]
            
            if len(available_columns) > 1:
                correlation_data = color_metrics[available_columns].corr()
                
                plt.figure(figsize=(12, 10))
                
                # Crear máscara triangular
                mask = np.triu(np.ones_like(correlation_data, dtype=bool))
                
                sns.heatmap(correlation_data, annot=True, cmap='RdBu_r', center=0,
                           square=True, fmt='.3f', mask=mask,
                           cbar_kws={"shrink": .8, "label": "Coeficiente de Correlación"},
                           annot_kws={"size": 14})
                
                plt.title('SwinIR: Matriz de Correlación de Métricas y Factores\n(Solo Imágenes a Color)', 
                         fontweight='bold', pad=25)
                plt.xlabel('Métricas')
                plt.ylabel('Métricas')
                
                # Agregar explicación en español
                explanation = ("Interpretación:\n"
                              "• +1: Correlación positiva perfecta\n"
                              "• 0: Sin correlación lineal\n"
                              "• -1: Correlación negativa perfecta\n"
                              "• |r| > 0.7: Correlación fuerte\n"
                              "• |r| < 0.3: Correlación débil")
                
                plt.text(1.15, 0.5, explanation, transform=plt.gca().transAxes,
                        verticalalignment='center', fontsize=14,
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "swinir_correlation_matrix.png"), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Guardar matriz de correlación
                correlation_data.to_csv(os.path.join(output_dir, "swinir_correlation_matrix.csv"))
                print("📊 Matriz de correlación guardada")
        
        # Análisis de eficiencia específico para SwinIR
        if timing_df is not None and metrics_df is not None:
            self._create_efficiency_analysis_swinir(metrics_df, timing_df, output_dir)
        
        # Restaurar configuración por defecto
        plt.rcdefaults()
    
    def _create_efficiency_analysis_swinir(self, metrics_df, timing_df, output_dir):
        """Crea análisis de eficiencia específico para SwinIR"""
        
        # Configurar fuente Computer Modern
        plt.rcParams.update({
            'font.family': ['serif'],
            'font.serif': ['Computer Modern Roman', 'Times', 'DejaVu Serif'],
            'font.size': 16,
            'axes.titlesize': 20,
            'axes.labelsize': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
            'legend.fontsize': 16
        })
        
        # Combinar métricas y timing
        color_metrics = metrics_df[metrics_df['image_type'] == 'color']
        
        # Análisis separado para GPU y CPU
        devices = ['gpu', 'cpu']
        device_names = {'gpu': 'GPU', 'cpu': 'CPU'}
        
        for device in devices:
            device_timing = timing_df[timing_df['device'] == device]
            
            if len(device_timing) == 0:
                print(f"⚠️  No hay datos de timing para {device.upper()}")
                continue
            
            # Calcular promedios por modelo
            metrics_avg = color_metrics.groupby('model_name')[self.metrics + ['scale_factor', 'input_resolution']].mean().reset_index()
            timing_avg = device_timing.groupby('model_name')[['mean_time_ms', 'fps']].mean().reset_index()
            
            # Combinar datos
            efficiency_data = pd.merge(metrics_avg, timing_avg, on='model_name', how='inner')
            
            if len(efficiency_data) == 0:
                print(f"⚠️  No hay datos combinados para {device.upper()}")
                continue
            
            # Gráfico de eficiencia: PSNR vs Tiempo
            plt.figure(figsize=(14, 10))
            
            scatter = plt.scatter(efficiency_data['mean_time_ms'], efficiency_data['psnr'], 
                                s=150, alpha=0.7, c=efficiency_data['ssim'], 
                                cmap='viridis', edgecolors='black', linewidth=0.8)
            
            for i, model in enumerate(efficiency_data['model_name']):
                plt.annotate(model, (efficiency_data['mean_time_ms'].iloc[i], 
                            efficiency_data['psnr'].iloc[i]),
                           xytext=(8, 8), textcoords='offset points', fontsize=12,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            plt.xlabel('Tiempo de Inferencia (ms)')
            plt.ylabel('PSNR (dB)')
            plt.title(f'SwinIR: Compromiso Calidad vs Velocidad (Inferencia {device_names[device]})', 
                     fontweight='bold', pad=25)
            
            # Colorbar con mejor formato
            cbar = plt.colorbar(scatter, label='SSIM')
            cbar.ax.tick_params(labelsize=14)
            
            plt.grid(True, alpha=0.3)
            
            # Agregar información en español
            info_text = f"Dispositivo: {device_names[device]}\nModelos: {len(efficiency_data)}\n"
            info_text += f"Rango de tiempo: {efficiency_data['mean_time_ms'].min():.1f}-{efficiency_data['mean_time_ms'].max():.1f} ms\n"
            info_text += f"Rango PSNR: {efficiency_data['psnr'].min():.2f}-{efficiency_data['psnr'].max():.2f} dB"
            
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontsize=14, 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"swinir_efficiency_analysis_{device}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Guardar datos de eficiencia
            efficiency_data.to_csv(os.path.join(output_dir, f"swinir_efficiency_data_{device}.csv"), index=False)
            print(f"📈 Análisis de eficiencia SwinIR {device.upper()} guardado")
            
            # Análisis adicional: tiempo por píxel procesado (solo para GPU)
            if device == 'gpu':
                efficiency_data['pixels_processed'] = efficiency_data['input_resolution'] ** 2
                efficiency_data['ms_per_megapixel'] = (efficiency_data['mean_time_ms'] / 
                                                     (efficiency_data['pixels_processed'] / 1e6))
                
                # Gráfico de escalabilidad
                plt.figure(figsize=(12, 8))
                plt.scatter(efficiency_data['input_resolution'], efficiency_data['ms_per_megapixel'],
                           s=100, alpha=0.7, c=efficiency_data['scale_factor'], cmap='plasma')
                
                for i, model in enumerate(efficiency_data['model_name']):
                    plt.annotate(model, (efficiency_data['input_resolution'].iloc[i], 
                                efficiency_data['ms_per_megapixel'].iloc[i]),
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
                
                plt.xlabel('Resolución de Entrada (píxeles)')
                plt.ylabel('Tiempo de Procesamiento por Megapíxel (ms/MP)')
                plt.title('SwinIR: Análisis de Escalabilidad Computacional', fontweight='bold')
                plt.colorbar(label='Factor de Escala')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "swinir_scalability_analysis.png"), dpi=300)
                plt.close()
        
        # Restaurar configuración por defecto
        plt.rcdefaults()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Análisis de Resultados SwinIR")
    parser.add_argument("--evaluation_dir", default="evaluation_results", 
                       help="Directorio con resultados de evaluación SwinIR")
    parser.add_argument("--output_dir", default="swinir_comprehensive_report",
                       help="Directorio para guardar reporte")
    
    args = parser.parse_args()
    
    # Verificar que existe el directorio
    if not os.path.exists(args.evaluation_dir):
        print(f"❌ Error: Directorio no encontrado: {args.evaluation_dir}")
        return 1
    
    print("📊 SISTEMA DE ANÁLISIS DE RESULTADOS SWINIR")
    print("=" * 50)
    print(f"Directorio de evaluación: {args.evaluation_dir}")
    print(f"Directorio de salida: {args.output_dir}")
    
    try:
        # Crear analizador
        analyzer = SwinIRResultsAnalyzer(args.evaluation_dir)
        
        # Generar reporte completo
        analyzer.generate_comprehensive_report(args.output_dir)
        
        print("\n✅ Análisis SwinIR completado exitosamente!")
        
    except Exception as e:
        print(f"\n💥 Error durante el análisis: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())