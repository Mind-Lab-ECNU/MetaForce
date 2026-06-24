#!/usr/bin/env python3
"""
恢复备份的数据文件

功能：
    将所有 .json.bak 恢复为 .json
    将所有 .parquet.bak 恢复为 .parquet

用法：
    python restore_backup.py

示例：
    train.json.bak -> train.json
    train.parquet.bak -> train.parquet
"""

import shutil
from pathlib import Path


# 数据目录路径（远程服务器）
DATA_DIR = Path("/inspire/hdd/project/ai4education/public/wsa_1.0/verltools/verl_m/data_multi_category/data")

# 是否删除备份文件（恢复后）
DELETE_BACKUP = False


def find_backup_files(data_dir: Path):
    """查找所有备份文件"""
    json_backups = []
    parquet_backups = []
    
    for bak_file in data_dir.rglob("*.bak"):
        # 跳过非目标备份文件
        if bak_file.name.endswith(".json.bak"):
            json_backups.append(bak_file)
        elif bak_file.name.endswith(".parquet.bak"):
            parquet_backups.append(bak_file)
    
    return sorted(json_backups), sorted(parquet_backups)


def find_and_remove_corrupted_parquet_files(data_dir: Path) -> int:
    """
    查找并删除损坏的 parquet 文件（如 train.parquet.as.json）
    只搜索特定模式，跳过 images 目录以提高性能
    
    Returns:
        删除的文件数量
    """
    corrupted_files = []
    
    # 直接搜索包含 .parquet. 的文件模式（跳过 images 目录）
    # 损坏的文件通常形如：train.parquet.as.json
    for pattern in ["*.parquet.*", "*.parquet.as.*"]:
        for file_path in data_dir.rglob(pattern):
            # 跳过 images 目录下的文件（不可能是损坏的 parquet）
            if "images" in str(file_path):
                continue
            # 跳过正常的 .parquet.bak 备份文件
            if file_path.name.endswith(".parquet.bak"):
                continue
            if file_path.is_file():
                corrupted_files.append(file_path)
    
    if not corrupted_files:
        return 0
    
    print(f"\n⚠️  发现 {len(corrupted_files)} 个损坏的 parquet 文件:")
    for f in corrupted_files:
        rel_path = f.relative_to(data_dir)
        print(f"  - {rel_path}")
    
    print(f"\n正在删除...")
    deleted_count = 0
    for f in corrupted_files:
        try:
            f.unlink()
            rel_path = f.relative_to(data_dir)
            print(f"  ✅ 已删除: {rel_path}")
            deleted_count += 1
        except Exception as e:
            print(f"  ❌ 删除失败 {f.name}: {e}")
    
    return deleted_count


def restore_backup(backup_path: Path, delete_backup: bool = False) -> bool:
    """
    恢复单个备份文件
    
    Args:
        backup_path: 备份文件路径
        delete_backup: 恢复后是否删除备份文件
        
    Returns:
        是否成功
    """
    # 计算原始文件路径
    if backup_path.name.endswith(".json.bak"):
        original_path = backup_path.with_suffix("").with_suffix(".json")
    elif backup_path.name.endswith(".parquet.bak"):
        original_path = backup_path.with_suffix("").with_suffix(".parquet")
    else:
        print(f"  ⚠️  未知备份格式: {backup_path.name}")
        return False
    
    try:
        # 恢复：将备份文件复制为原始文件
        shutil.copy2(backup_path, original_path)
        print(f"  ✅ {backup_path.name} -> {original_path.name}")
        
        # 如果需要，删除备份文件
        if delete_backup:
            backup_path.unlink()
            print(f"  🗑️  已删除备份: {backup_path.name}")
        
        return True
    except Exception as e:
        print(f"  ❌ 恢复失败: {e}")
        return False


def main():
    """主函数"""
    print("=" * 60)
    print("恢复备份数据文件")
    print("=" * 60)
    print(f"数据目录: {DATA_DIR}")
    print(f"删除备份: {'是' if DELETE_BACKUP else '否'}")
    print()
    
    # 第一步：删除损坏的 parquet 文件
    print("【步骤 1/3】清理损坏的 parquet 文件...")
    deleted_count = find_and_remove_corrupted_parquet_files(DATA_DIR)
    if deleted_count > 0:
        print(f"✅ 已清理 {deleted_count} 个损坏文件\n")
    else:
        print("✅ 未发现损坏文件\n")
    
    # 第二步：查找所有备份文件
    print("【步骤 2/3】扫描备份文件...")
    json_backups, parquet_backups = find_backup_files(DATA_DIR)
    
    print(f"  找到 {len(json_backups)} 个 JSON 备份文件")
    print(f"  找到 {len(parquet_backups)} 个 Parquet 备份文件")
    
    if not json_backups and not parquet_backups:
        print("\n没有找到备份文件，程序退出")
        return
    
    # 第三步：恢复 JSON 备份
    if json_backups:
        print(f"\n【步骤 3/3】恢复 JSON 备份文件 ({len(json_backups)} 个)")
        print("-" * 60)
        json_success = 0
        for backup_file in json_backups:
            rel_path = backup_file.relative_to(DATA_DIR)
            print(f"\n处理: {rel_path}")
            if restore_backup(backup_file, DELETE_BACKUP):
                json_success += 1
    else:
        json_success = 0
    
    # 恢复 Parquet 备份（也在步骤3中）
    if parquet_backups:
        print(f"\n继续恢复 Parquet 备份文件 ({len(parquet_backups)} 个)")
        print("-" * 60)
        parquet_success = 0
        for backup_file in parquet_backups:
            rel_path = backup_file.relative_to(DATA_DIR)
            print(f"\n处理: {rel_path}")
            if restore_backup(backup_file, DELETE_BACKUP):
                parquet_success += 1
    else:
        parquet_success = 0
    
    # 总结
    print("\n" + "=" * 60)
    print("恢复完成！")
    print("=" * 60)
    print(f"JSON 备份: {json_success}/{len(json_backups)} 个成功")
    print(f"Parquet 备份: {parquet_success}/{len(parquet_backups)} 个成功")
    if DELETE_BACKUP:
        print("备份文件已删除")
    else:
        print("备份文件保留（可手动删除或设置 DELETE_BACKUP=True）")
    print("=" * 60)


if __name__ == "__main__":
    main()
