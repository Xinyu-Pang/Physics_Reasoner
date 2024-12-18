import re
import pdb
import ast
import argparse
from fractions import Fraction

def extract_formula_content(formulas):
    """Extract formula basic contents for LLM prompting."""
    formulas_contents = [{'name':formulas[id]['name'], 'content': formulas[id]['content']} for id in formulas]
    return formulas_contents

def convert_str2list(input_str):
    """Convert string to list."""
    input_str = input_str.replace("\n", "")
    input_str = re.sub(r'-?\d+\/\d+', replace_fraction, input_str)
    matches = re.search(r"\[.*\]", input_str)
    if matches: 
        try:
            extracted_list_direct = ast.literal_eval(matches.group(0))
            return list(extracted_list_direct)
        except:
            pdb.set_trace()
            return "None"
    else: 
        return "None"
    
    
def replace_fraction(match):
    fraction = match.group(0)
    return str(eval(fraction))

def process_code(input_code):
    """处理LLM生成的Python代码"""
    code = input_code.replace("python\n", "").replace("plaintext", "")
    # code = code.replace("'", "")
    match = code.split('```')
    if len(match) >= 3:    
        out_code = match[-2]
        # pdb.set_trace()
        return out_code
    else:
        pdb.set_trace()
        return ""

def process_review(input_code):
    """处理LLM生成的Python代码"""
    code = input_code.replace("python\n", "").replace("plaintext", "")
    # code = code.replace("'", "")
    match = code.split('```')
    if len(match) >= 3:    
        out_code = match[-2]
        # pdb.set_trace()
    else:
        pdb.set_trace()
        out_code =  ""
    review = "```".join(match[:-2])
    return review, out_code

def cal_not(inputs):
    """计算给定数字和幂的乘积
    如果出现错误，会返回原始输入字符串"""
    try:
        x,ab=list(inputs)
        match_number = re.compile('10\^[{]?\ *-?[0-9]+\ *[}]?')
        ab=re.findall(match_number, ab)[0]
        ab=ab[ab.find('^')+1:]
        if '{' in ab:
            ab=ab[ab.find('{')+1:]
        if '}' in ab:
            ab=ab[:ab.find('}')]
        x=x.strip()
        out=float(x)*10**float(ab)
        # print(float(x)*10**float(ab))
        return str(out)
    except:
        print('error')
    return inputs

def process_unit(input_string):
    # 如果存在10的次幂，去除
    if remove_not(input_string):
        input_string = remove_not(input_string)
    
    # 将latex符号，换成可读文本符号
    u = input_string.replace('^{\\circ}', 'degree').replace('\mu ','u').replace('$', '')
    u = u.replace('\cdot', '*').replace('electrons', '').replace(';', '')

    # 去除'\mathrm{}'
    pattern_rm = re.compile(r'\\mathrm\{([^}]*)\}')
    u = re.sub(pattern_rm, replace, u)
    
    # 去除'\hat{}'【及其内部内容】
    pattern_hat = re.compile(r'\\hat\{([^}]*)\}')
    u = pattern_hat.sub('', u)

    # 去除'\text{}'
    pattern_txt = re.compile(r'\\text\s*\{([^}]*)\}')
    u = re.sub(pattern_txt, replace, u)

    return u

def remove_not(x):
    """从字符串中移除形如"$10^{x}$"的表达式
    如果找到了，会返回去除后的格式
    如果没找到, 返回None"""
    match_number = re.compile('[\$]?\ *10\^[{]?\ *-?[0-9]+\ *[}]?\ *[\$]?')
    result=re.findall(match_number, x)
    if len(result) !=0:
        return re.split(match_number, x)[-1] + ' '
    return None

# 定义一个替换函数，用于替换匹配到的\mathrm{}为其中的内容
def replace(match):
    return match.group(1).replace("~","")

# 用于reason.py输入bool参数
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# 来自eval
def remove_boxed(s):
    match = re.search(r'\\boxed{([^}]*)}', s)
    if match:
        return match.group(1)
    else:
        return ''
    
def process_output(s):
    try:
        pred_ans = float(s)
    except:
        try:
            pred_ans = float(Fraction(s))
        except:
            matches = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
            if matches:
                pred_ans = float(matches[0])
            else:
                pred_ans = 'No matching number: {}'.format(s)
    return pred_ans