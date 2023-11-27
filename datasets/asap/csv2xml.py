from xml.etree.ElementTree import Element, ElementTree, SubElement
import csv
import os


def csv_to_xml(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)

        root = Element("sentences")
        d = {
            '1': 'positive',
            '0': 'neutral',
            '-1': 'negative',
            '-2': 'NULL'
        }

        for row in reader:
            # for index, column in enumerate(row):
            node_sentence = Element('sentence')
            root.append(node_sentence)
            node_text = SubElement(node_sentence, 'text')
            # 给叶子节点text设置一个文本节点，用于显示文本内容
            node_text.text = row[1]
            node_aspectCategories = Element("aspectCategories")
            node_sentence.append(node_aspectCategories)
            node_aspectCategory = Element("aspectCategory", {'category': 'Location#Transportation', 'polarity': d[row[3]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Location#Downtown', 'polarity': d[row[4]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Location#Easy_to_find', 'polarity': d[row[5]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Service#Queue', 'polarity': d[row[6]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Service#Hospitality', 'polarity': d[row[7]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Service#Parking', 'polarity': d[row[8]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Service#Timely', 'polarity': d[row[9]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Price#Level', 'polarity': d[row[10]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Price#Cost_effective', 'polarity': d[row[11]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Price#Discount', 'polarity': d[row[12]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Ambience#Decoration', 'polarity': d[row[13]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Ambience#Noise', 'polarity': d[row[14]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Ambience#Space', 'polarity': d[row[15]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Ambience#Sanitary', 'polarity': d[row[16]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Food#Portion', 'polarity': d[row[17]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Food#Taste', 'polarity': d[row[18]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Food#Appearance', 'polarity': d[row[19]]})
            node_aspectCategories.append(node_aspectCategory)
            node_aspectCategory = Element("aspectCategory", {'category': 'Food#Recommend', 'polarity': d[row[20]]})
            node_aspectCategories.append(node_aspectCategory)
    beatau(root)
    return ElementTree(root)


def beatau(e, level=0):
    if len(e) > 0:
        e.text = '\n' + '\t' * (level + 1)
        for child in e:
            beatau(child, level + 1)
        child.tail = child.tail[:-1]
    e.tail = '\n' + '  ' * level

    # 开始写xml文档
    # with open(path, "w", encoding="utf-8") as f:
    #     # writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
    #     # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
    #     doc.writexml(f, indent='', addindent='\t', newl='\n', encoding="utf-8")


if __name__ == '__main__':
    base_path = os.path.join('xml_file')
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    raw_train_path = os.path.join(base_path, 'raw/train.xml')
    raw_val_path = os.path.join(base_path, 'raw/val.xml')
    raw_test_path = os.path.join(base_path, 'raw/test.xml')

    raw_train_sample_path = os.path.join(base_path, 'raw/train_sample.xml')
    raw_val_sample_path = os.path.join(base_path, 'raw/val_sample.xml')
    raw_test_sample_path = os.path.join(base_path, 'raw/test_sample.xml')


    train = '/home/jye/ASAP/data/train.csv'
    ET = csv_to_xml(train)
    ET.write(raw_train_path, encoding='utf-8')

    val = '/home/jye/ASAP/data/dev.csv'
    ET = csv_to_xml(val)
    ET.write(raw_val_path, encoding='utf-8')

    test = '/home/jye/ASAP/data/test.csv'
    ET = csv_to_xml(test)
    ET.write(raw_test_path, encoding='utf-8')

    train_sample = '/home/jye/ASAP/data/train_sample.csv'
    ET = csv_to_xml(train_sample)
    ET.write(raw_train_sample_path, encoding='utf-8')

    val_sample = '/home/jye/ASAP/data/dev_sample.csv'
    ET = csv_to_xml(val_sample)
    ET.write(raw_val_sample_path, encoding='utf-8')

    test_sample = '/home/jye/ASAP/data/test_sample.csv'
    ET = csv_to_xml(test_sample)
    ET.write(raw_test_sample_path, encoding='utf-8')
