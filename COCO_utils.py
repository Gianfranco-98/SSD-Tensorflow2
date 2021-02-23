#!/usr/bin/env python3

from pycocotools.coco import COCO


def get_image_content(coco, ID):
    content = []
    tmp_list = []
    for ann_dict in coco.anns.items():
        if ann_dict[1]['image_id'] == ID:
            if ann_dict[1]['category_id'] not in tmp_list:
                tmp_list.append(ann_dict[1]['category_id'])
                element = {
                    'object': ann_dict[1]['category_id'], 
                    'count': 1, 
                    'area': ann_dict[1]['area'], 
                    'bbox': ann_dict[1]['bbox']}
                content.append(element)
            else:
                for i in range(len(tmp_list)):
                    if ann_dict[1]['category_id'] == tmp_list[i]:
                        content[i]['count'] += 1
                        if content[i]['count'] > 2:
                            content[i]['area'].append(ann_dict[1]['area'])
                            content[i]['bbox'].append(ann_dict[1]['bbox'])
                        else:
                            content[i]['area'] = [content[i]['area'], ann_dict[1]['area']]
                            content[i]['bbox'] = [content[i]['bbox'], ann_dict[1]['bbox']]
    return content


def global_info(coco):
    urls = []
    ids = coco.getImgIds()
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        urls.append(img_meta['coco_url'])
    labels = [coco.cats[id]['name'] for id in coco.getCatIds()]
    return ids, urls, labels