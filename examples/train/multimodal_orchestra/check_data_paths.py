#!/usr/bin/env python3
"""
检查训练脚本中配置的数据集路径是否存在

用法:
  # 本地模式（直接在服务器上运行）
  python check_data_paths.py --local

  # 远程模式（通过 SSH 检查远程服务器）
  python check_data_paths.py --host user@server_hostname

  # 指定脚本路径
  python check_data_paths.py --script /path/to/train_8gpu.sh --local
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def extract_shell_variable(script_content, var_name):
    """从 shell 脚本中提取变量值"""
    # 匹配 VAR_NAME="value" 格式
    pattern = rf'{var_name}="([^"]+)"'
    match = re.search(pattern, script_content)
    if match:
        return match.group(1)
    return None


def parse_array_paths(script_content, array_name, data_root):
    """解析 shell 数组中的路径"""
    # 匹配数组定义，如: train_data_id_core=( ... )
    pattern = rf'{array_name}=\((.*?)\)'
    match = re.search(pattern, script_content, re.DOTALL)
    
    paths = []
    if match:
        array_content = match.group(1)
        # 提取所有 ${DATA_ROOT} 开头的路径
        path_pattern = r'\$\{DATA_ROOT\}([^"\s\]]+)'
        for m in re.finditer(path_pattern, array_content):
            relative_path = m.group(1)
            full_path = data_root + relative_path
            paths.append(full_path)
    
    return paths


def check_local_path_exists(path):
    """检查本地路径是否存在"""
    return os.path.exists(path)


def check_remote_path_exists(host, path):
    """通过 SSH 检查远程服务器上的路径是否存在"""
    try:
        result = subprocess.run(
            ['ssh', host, f'test -e "{path}" && echo "EXISTS" || echo "NOT_FOUND"'],
            capture_output=True,
            text=True,
            timeout=30
        )
        return "EXISTS" in result.stdout
    except subprocess.TimeoutExpired:
        return "ERROR: SSH 连接超时"
    except FileNotFoundError:
        return "ERROR: 未找到 ssh 命令"
    except Exception as e:
        return f"ERROR: {e}"


def get_file_size(host, path, local_mode):
    """获取文件大小"""
    try:
        if local_mode:
            size = os.path.getsize(path)
        else:
            result = subprocess.run(
                ['ssh', host, f'stat -c%s "{path}" 2>/dev/null || echo "0"'],
                capture_output=True,
                text=True,
                timeout=10
            )
            size = int(result.stdout.strip())
        
        # 转换为人类可读格式
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f}KB"
        elif size < 1024 * 1024 * 1024:
            return f"{size/(1024*1024):.1f}MB"
        else:
            return f"{size/(1024*1024*1024):.2f}GB"
    except:
        return "Unknown"


def main():
    parser = argparse.ArgumentParser(
        description='检查训练数据路径是否存在',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python check_data_paths.py --local
  python check_data_paths.py --host user@server
  python check_data_paths.py --script train_8gpu.sh --local
        """
    )
    parser.add_argument(
        '--script',
        type=str,
        default='single_node/train_8gpu.sh',
        help='训练脚本的路径 (默认: single_node/train_8gpu.sh)'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='本地模式：直接在本地检查文件是否存在'
    )
    parser.add_argument(
        '--host',
        type=str,
        default=None,
        help='远程服务器地址，如 user@hostname (用于远程检查)'
    )
    parser.add_argument(
        '--show-size',
        action='store_true',
        help='显示文件大小'
    )
    
    args = parser.parse_args()
    
    # 如果没有指定模式，提示用户
    if not args.local and args.host is None:
        print("错误: 请指定 --local 或 --host user@server")
        parser.print_help()
        sys.exit(1)
    
    # 确定脚本路径
    script_dir = Path(__file__).parent
    if os.path.isabs(args.script):
        script_path = args.script
    else:
        script_path = script_dir / args.script
    
    if not os.path.exists(script_path):
        print(f"错误: 找不到脚本文件: {script_path}")
        sys.exit(1)
    
    # 读取脚本内容
    with open(script_path, 'r') as f:
        script_content = f.read()
    
    # 提取 DATA_ROOT
    data_root = extract_shell_variable(script_content, 'DATA_ROOT')
    if not data_root:
        print("错误: 无法从脚本中找到 DATA_ROOT 变量")
        sys.exit(1)
    
    print("=" * 100)
    print(f"脚本文件: {script_path}")
    print(f"DATA_ROOT: {data_root}")
    print(f"检查模式: {'本地' if args.local else f'远程 ({args.host})'}")
    print("=" * 100)
    
    # 定义要检查的数据集数组
    all_datasets = [
        ('train_data_id_core', '训练数据 (ID核心 - 10个)'),
        ('train_data_id_aux', '训练数据 (ID辅助 - 10个)'),
        ('val_data_id', '验证数据 (ID测试 - 10个)'),
        ('val_data_ood', '验证数据 (OOD测试 - 10个)'),
    ]
    
    # 检查函数
    check_func = check_local_path_exists if args.local else lambda p: check_remote_path_exists(args.host, p)
    
    total_paths = 0
    existing_paths = 0
    not_found_paths = []
    
    for array_name, description in all_datasets:
        print(f"\n【{description}】")
        paths = parse_array_paths(script_content, array_name, data_root)
        
        if not paths:
            print(f"  警告: 未找到 {array_name} 的路径")
            continue
        
        for path in paths:
            total_paths += 1
            exists = check_func(path)
            
            size_info = ""
            if args.show_size and exists is True:
                size_str = get_file_size(args.host, path, args.local)
                size_info = f" [{size_str}]"
            
            if exists is True:
                status = "✓ 存在"
                existing_paths += 1
            elif exists is False:
                status = "✗ 不存在"
                not_found_paths.append((description, path))
            else:
                status = f"? {exists}"
                not_found_paths.append((description, path))
            
            print(f"  {status}: {path}{size_info}")
    
    # 汇总结果
    print("\n" + "=" * 100)
    print(f"总计路径数: {total_paths}")
    print(f"存在: {existing_paths} 个")
    print(f"缺失: {len(not_found_paths)} 个")
    
    if existing_paths == total_paths:
        print("\n✓ 所有路径都存在！")
    
    if not_found_paths:
        print("\n缺失的路径列表:")
        for category, path in not_found_paths:
            print(f"  [{category}] {path}")
        sys.exit(1)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
