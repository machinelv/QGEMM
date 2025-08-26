import sys
import re
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from collections import defaultdict

def parse_input():
    """从标准输入解析测试结果"""
    input_text = sys.stdin.read()
    
    # 按分隔线分割不同的测试
    sections = input_text.split("====================================================================================================")
    
    tests = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        lines = section.split('\n')
        if not lines:
            continue
            
        # 第一行是指令
        instruction_line = None
        for line in lines:
            if line.startswith("Instruction:"):
                instruction_line = line
                break
        
        if not instruction_line:
            continue
            
        instruction = instruction_line.replace("Instruction:", "").strip()
        
        # 解析线程数据
        thread_data = {}
        for line in lines:
            if line.startswith("thread"):
                # 解析线程号和持有的数据
                match = re.match(r'thread (\d+) holds: (.+)', line)
                if match:
                    thread_id = int(match.group(1))
                    data_str = match.group(2)
                    
                    # 解析数据项，格式如: (0) 0, 1,(1) 64, 65,(2) 128, 129,(3) 192, 193,
                    # 或者: (0) 0, 8,(1) 64, 72,(2) 128, 136,(3) 192, 200,
                    # 每个寄存器包含两个16位数据的内存位置
                    position_register_pairs = []
                    
                    # 使用更灵活的正则表达式提取寄存器编号和对应的位置对
                    # 匹配 (寄存器编号) 后面跟着的两个数字，数字之间用逗号和可选空格分隔
                    pattern = r'\((\d+)\)\s*(\d+)\s*,\s*(\d+)'
                    matches = re.findall(pattern, data_str)
                    
                    for reg_id_str, pos1_str, pos2_str in matches:
                        reg_id = int(reg_id_str)
                        try:
                            pos1 = int(pos1_str)
                            pos2 = int(pos2_str)
                            # 每个32位寄存器包含两个16位数据
                            position_register_pairs.append((pos1, reg_id))
                            position_register_pairs.append((pos2, reg_id))
                        except ValueError:
                            pass
                    
                    thread_data[thread_id] = position_register_pairs
        
        if thread_data:
            tests.append({
                'instruction': instruction,
                'thread_data': thread_data
            })
    
    return tests

def create_layout_map(thread_data, max_pos=256):
    """根据线程数据创建位置到线程和寄存器的映射"""
    pos_to_thread_reg = {}
    
    for thread_id, position_register_pairs in thread_data.items():
        for pos, reg_id in position_register_pairs:
            if pos < max_pos:
                pos_to_thread_reg[pos] = (thread_id, reg_id)
    
    return pos_to_thread_reg

def generate_visualization(test_data):
    """为每个测试生成可视化"""
    # 构建颜色映射（32个不同颜色）
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    thread_colors = {t: colors[t % len(colors)] for t in range(32)}
    
    for i, test in enumerate(test_data):
        instruction = test['instruction']
        thread_data = test['thread_data']
        
        # 创建位置映射
        pos_to_thread_reg = create_layout_map(thread_data)
        
        # 创建16x16表格
        fig, ax = plt.subplots(figsize=(16, 8))
        
        table_data = []
        cell_colors = []
        
        for row in range(16):
            row_vals = []
            row_cols = []
            for col in range(16):
                pos = row * 16 + col
                if pos in pos_to_thread_reg:
                    thread_id, reg_id = pos_to_thread_reg[pos]
                    label = f"T{thread_id}R{reg_id}P{pos}"
                    color = thread_colors[thread_id]
                else:
                    label = f"P{pos}"
                    color = 'lightgray'
                
                row_vals.append(label)
                row_cols.append(color)
            
            table_data.append(row_vals)
            cell_colors.append(row_cols)
        
        # 创建表格
        the_table = ax.table(
            cellText=table_data,
            cellColours=cell_colors,
            loc="center",
            cellLoc="center",
        )
        
        the_table.auto_set_font_size(False)
        the_table.set_fontsize(18)
        the_table.scale(1.2, 4)
        
        ax.axis("off")
        plt.title(f"Memory Layout: {instruction}", fontsize=28, pad=20)
        
        # 保存图片
        safe_filename = re.sub(r'[^\w\-_\.]', '_', instruction)
        img_path = f"./output/{safe_filename}.png"
        plt.savefig(img_path, bbox_inches="tight", dpi=150)
        plt.close()
        
        print(f"Generated visualization for '{instruction}' -> {img_path}")

def generate_markdown_report(test_data):
    """生成markdown格式的报告以检查解析结果"""
    markdown_content = []
    markdown_content.append("# Layout Analysis Report\n")
    markdown_content.append(f"Generated on: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    markdown_content.append(f"Total instructions analyzed: {len(test_data)}\n")
    
    intruction_name = ""
    for i, test in enumerate(test_data, 1):
        instruction = test['instruction']
        thread_data = test['thread_data']
        intruction_name = instruction.split(".")[0]
        
        markdown_content.append(f"\n## {i}. {instruction}\n")
        
        # 统计信息
        total_threads = len(thread_data)
        total_positions = sum(len(position_register_pairs) for position_register_pairs in thread_data.values())
        
        markdown_content.append(f"- **Threads involved**: {total_threads}")
        markdown_content.append(f"- **Total positions**: {total_positions}")
        
        # 每个线程的详细信息
        markdown_content.append(f"\n### Thread Details:\n")
        
        for thread_id in sorted(thread_data.keys()):
            position_register_pairs = thread_data[thread_id]
            formatted_data = [(pos, reg) for pos, reg in position_register_pairs]
            markdown_content.append(f"- **Thread {thread_id}**: {len(position_register_pairs)} positions -> {formatted_data}")
        
        # 位置分布统计
        pos_to_thread_reg = create_layout_map(thread_data)
        markdown_content.append(f"\n### Position Distribution (first 64 positions):\n")
        markdown_content.append("```")
        for row in range(8):  # 显示前8行
            row_info = []
            for col in range(8):  # 显示前8列
                pos = row * 8 + col
                if pos in pos_to_thread_reg:
                    thread_id, reg_id = pos_to_thread_reg[pos]
                    row_info.append(f"T{thread_id:2d}R{reg_id}")
                else:
                    row_info.append("  ----")
            markdown_content.append(" ".join(row_info))
        markdown_content.append("```")
        
        # 线程覆盖率分析
        coverage = {}
        for pos in range(256):
            if pos in pos_to_thread_reg:
                thread_id, reg_id = pos_to_thread_reg[pos]
                if thread_id not in coverage:
                    coverage[thread_id] = []
                coverage[thread_id].append(pos)
        
        markdown_content.append(f"\n### Coverage Analysis:")
        markdown_content.append(f"- **Covered positions**: {len(pos_to_thread_reg)}/256 ({len(pos_to_thread_reg)/256*100:.1f}%)")
        markdown_content.append(f"- **Threads with coverage**: {len(coverage)}")
        
        if len(coverage) > 0:
            avg_positions_per_thread = len(pos_to_thread_reg) / len(coverage)
            markdown_content.append(f"- **Average positions per thread**: {avg_positions_per_thread:.1f}")
        
        markdown_content.append("\n---")
    
    # 写入markdown文件
    markdown_filename = f"./output/report_{intruction_name}.md"
    with open(markdown_filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_content))
    
    print(f"Generated markdown report: {markdown_filename}")
    return markdown_filename

def main():
    try:
        # 从管道读取输入
        test_data = parse_input()
        
        if not test_data:
            print("No valid test data found in input", file=sys.stderr)
            sys.exit(1)
        
        print(f"Found {len(test_data)} test instructions:")
        for test in test_data:
            print(f"  - {test['instruction']}")
        
        # 生成markdown报告
        generate_markdown_report(test_data)
        
        # 生成可视化
        generate_visualization(test_data)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()