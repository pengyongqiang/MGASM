import json
import pickle
import random
import re

# 简化的国家名列表
countries = [
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola", "Antigua and Barbuda", "Argentina", "Armenia",
    "Australia", "Austria",
    "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh", "Barbados", "Belarus", "Belgium", "Belize", "Benin", "Bhutan",
    "Bolivia",
    "Bosnia and Herzegovina", "Botswana", "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi", "Cabo Verde",
    "Cambodia", "Cameroon",
    "Canada", "Central African Republic", "Chad", "Chile", "China", "Colombia", "Comoros", "Congo", "Costa Rica",
    "Croatia", "Cuba",
    "Cyprus", "Czech Republic", "Denmark", "Djibouti", "Dominica", "Dominican Republic", "East Timor", "Ecuador",
    "Egypt", "El Salvador",
    "Equatorial Guinea", "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji", "Finland", "France", "Gabon", "Gambia",
    "Georgia", "Germany",
    "Ghana", "Greece", "Grenada", "Guatemala", "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras", "Hungary",
    "Iceland", "India",
    "Indonesia", "Iran", "Iraq", "Ireland", "Israel", "Italy", "Ivory Coast", "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya",
    "Kiribati", "Korea, North", "Korea, South", "Kosovo", "Kuwait", "Kyrgyzstan", "Laos", "Latvia", "Lebanon",
    "Lesotho", "Liberia",
    "Libya", "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar", "Malawi", "Malaysia", "Maldives", "Mali",
    "Malta", "Marshall Islands",
    "Mauritania", "Mauritius", "Mexico", "Micronesia", "Moldova", "Monaco", "Mongolia", "Montenegro", "Morocco",
    "Mozambique", "Myanmar",
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand", "Nicaragua", "Niger", "Nigeria", "North Macedonia",
    "Norway", "Oman",
    "Pakistan", "Palau", "Panama", "Papua New Guinea", "Paraguay", "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania",
    "Russia", "Rwanda", "Saint Kitts and Nevis", "Saint Lucia", "Saint Vincent and the Grenadines", "Samoa",
    "San Marino", "Sao Tome and Principe",
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles", "Sierra Leone", "Singapore", "Slovakia", "Slovenia",
    "Solomon Islands", "Somalia",
    "South Africa", "South Sudan", "Spain", "Sri Lanka", "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tajikistan",
    "Tanzania", "Thailand", "Togo", "Tonga", "Trinidad and Tobago", "Tunisia", "Turkey", "Turkmenistan", "Tuvalu",
    "Uganda", "Ukraine",
    "United Arab Emirates", "United Kingdom", "United States", "Uruguay", "Uzbekistan", "Vanuatu", "Vatican City",
    "Venezuela", "Vietnam",
    "Yemen", "Zambia", "Zimbabwe", 'USA', 'US', 'UK', 'CA', 'UAE'
]


def find_all_positions(string, character):
    positions = []
    index = 0
    while index < len(string):
        position = string.find(character, index)
        if position == -1:
            break
        positions.append(position)
        index = position + 1
    return positions


def get_phone(anchor_flag, user_index, g1_len):
    phone_arr = ['华为Mate30', '苹果iPhone 11', '苹果iPhone 11', '华为Mate30 Pro',
                 'Redmi K20 Pro', '华为nova 5 Pro 5G',
                 '华为Mate30', '荣耀9X', 'vivo Y3',
                 'OPPO A9', '荣耀20', 'OPPO A11', 'OPPO Reno', '华为畅享10 Plus',
                 'vivo S1', '华为P30', 'vivo X27', '苹果iPhone 11 Pro', '苹果 iPhone 11 Pro',
                 '一加7T', 'OPPO Reno2', 'vivo Y7s',
                 '苹果 iPhone X', '苹果 iPhone 8', '苹果 iPhone plus', '苹果 iPhone 7',
                 '红米Note 7 Pro', '小米9', '红米Note 7',
                 '小米Mix 2s', '小米8', '红米Note 5A', 'iPhone XS',
                 '华为 p10' '华为 p30', '华为 Mate20 Pro', '小米8',
                 'oppo Find X', 'OPPO R15', '三星 Galaxy S9', 'vivo X21',
                 '红米5 Plus', '坚果R1', '荣耀Magic 2', '荣耀V20',
                 'iphone 客户端', 'iphone 客户端', 'iphone 客户端', '苹果 iPhone 11',
                 '苹果iPhone 11', '努比亚X', '一加6T', '努比亚Z17S',
                 '三星Note8', 'nova 4e', '华为麦芒8', 'nova 5', 'Mate 20 X 5G',
                 '红米Note 8 Pro', '三星Galaxy A50', 'realme X50', '魅族16s Pro',
                 '黑鲨游戏手机 Helo', '中兴 AXON 10 Pro', '华为 麦芒 8', '努比亚 红魔3',
                 'realme X', '联想 Z5s', '魅族 16Xs', '苹果 iPhone XR', '魅族 Note9',
                 'vivo Z3x', '摩托罗拉 P30 note', '红米 K20 Pro',
                 'vivo X27 Pro', 'OPPO R17', '苹果 iPhone 8Plus', '三星 Galaxy A60',
                 '华为 畅享 9S', '努比亚 红魔3', 'iPhone XS Max', 'iPhone', 'iPhone SE', '荣耀30 Pro',
                 '坚果手机 Pro', '华为手机 畅享玩不停']
    stop_words = ['手工',
                  '微博 weibo.com',
                  '微博电影', '微博 weibo.com',
                  '日常 · 视频社区',
                  '美食侦探 · 视频社区',
                  '华为手机 畅享玩不停', '微博国际版',
                  '生日动态', '分享按钮', '美拍', 'iOS', '翻唱改编 · 视频社区',
                  '微博 weibo.com', '微博视频', '旅行', '红包活动',
                  '微公益', '生日动态', '日常', '微博新鲜事', '微博国际版',
                  '微博国际版', 'iPhone 11', '微博视频', '王丘丘超话', '野路子iPhone客户端',
                  '风景旅拍', '关联博客', '生日动态', '微博国际版', '红包活动',
                  '歪果仁研究...', '网易云音乐', '手办绘画 · 视频社区', '超话']
    if not anchor_flag:
        array = phone_arr + stop_words
        phone = array[random.randint(0, len(array) - 1)]
        while user_index >= g1_len and '微博' in phone:
            phone = array[random.randint(0, len(array) - 1)]
        return phone
    else:
        return phone_arr[random.randint(0, len(phone_arr) - 1)]


if __name__ == '__main__':
    g1, g2 = pickle.load(open('../data/wd/networks', 'rb'))
    attrs = pickle.load(open('../data/wd/attrs_origin', 'rb'))
    anchors = dict(json.load(open('../data/wd/anchors.txt', 'r')))
    r_anchors = {value: key for key, value in anchors.items()}
    anchors.update(r_anchors)
    already_user = []
    for x_user_index in attrs:
        if x_user_index not in already_user:
            anchor_flag = anchors.get(x_user_index) is not None

            tag = get_phone(anchor_flag, x_user_index, len(g1))

            x_user = attrs[x_user_index]
            x_user_topic = x_user[2]
            x_position = find_all_positions(x_user_topic, '来自')
            x_user_tags = []
            for start in x_position:
                newline_index = x_user_topic.find('\n', start)
                if not (newline_index == -1 or newline_index - start > 15):
                    x_tag = x_user_topic[start:newline_index]
                    x_user_tags.append(x_tag)
            for x_tag in x_user_tags:
                x_user_topic = x_user_topic.replace(x_tag, '来自[' + tag + ']')
            else:
                x_user_topic = x_user_topic + '来自[' + tag + ']'
            attrs[x_user_index][2] = x_user_topic
            already_user.append(x_user_index)

            # 如果是锚定用户
            y_user_index = anchors.get(x_user_index)
            if anchor_flag:
                if random.random() < 0.15:
                    tag = get_phone(False, y_user_index, len(g1))
                # if random.random() < 0.2:
                #     # 在这里写下您想要以30%概率执行的代码
                #     tag = tag.split()
                #     tag = ' '.join(tag[0:random.randint(1, len(tag))])

                y_user = attrs[y_user_index]
                y_user_topic = y_user[2]
                y_position = find_all_positions(y_user_topic, '来自')
                y_user_tags = []
                for start in y_position:
                    newline_index = y_user_topic.find('\n', start)
                    if not (newline_index == -1 or newline_index - start > 15):
                        y_tag = y_user_topic[start:newline_index]
                        y_user_tags.append(y_tag)
                if len(y_user_tags) > 0:
                    for y_tag in y_user_tags:
                        y_user_topic = y_user_topic.replace(y_tag, '来自[' + tag + ']')
                else:
                    y_user_topic = y_user_topic + '来自[' + tag + ']'
                attrs[y_user_index][2] = y_user_topic
                already_user.append(y_user_index)

    pickle.dump(attrs, open('../data/wd/attrs', 'wb'))

# if __name__ == '__main__':
#     attrs = pickle.load(open('../data/wd/attrs', 'rb'))
#     print(len(attrs))
#     anchors = dict(json.load(open('../data/wd/anchors.txt', 'r')))
#     print(len(anchors))
#     # r_anchors = {value: key for key, value in anchors.items()}
#     # anchors.update(r_anchors)
#     # count = 0
#     # for x in attrs:
#     #     x_name = attrs[x][0]
#     #     for y in attrs:
#     #         y_essay = attrs[y][2]
#     #         if x_name in y_essay and anchors.get(x) is None:
#     #             y_essay.replace(x_name, '')
#     #             if random.random() < 0.8:
#     #                 attrs[y][2] = y_essay
#     #                 count += 1
#     #     print('x=', x)
#     # print('count=', count)
#     # pickle.dump(attrs, open('../data/wd/attrs_new', 'wb'))


# if __name__ == '__main__':
#     pickle.dump(countries, open('../data/dblp/countries', 'wb'))
#
#     attrs = pickle.load(open('../data/dblp/attrs', 'rb'))
#     anchors = dict(json.load(open('../data/dblp/anchors.txt', 'r')))
#     r_anchors = {value: key for key, value in anchors.items()}
#     anchors.update(r_anchors)
#     count = 0
#     for x in attrs:
#         x_name = attrs[x][0]
#         x_area = attrs[x][1]
#         x_essay = attrs[x][2]
#         country_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(country) for country in countries) + r')\b',
#                                      flags=re.IGNORECASE)
#         x_area = country_pattern.sub('', x_area)
#
#         print('x_name=', x_name)
#         print('x_area=', x_area)
