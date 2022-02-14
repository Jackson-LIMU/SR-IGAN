# Super-Resolution-Reconstruction-for-Photomicrographs-of-Coal-Based-on-Improved-Generative-Adversaria

煤炭作为主要化石燃料之一，在满足日益增长的能源需求方面发挥着不可或缺的作用，在可预见的未来仍将是全球能源体系的支柱。而我国能源富煤、贫油、少气的基本特点决定了短期内难以改变以煤炭为主体的能源结构。在采矿和地质领域正确识别煤炭的岩相特性至关重要。然而想要获取高质量的煤岩显微组分图片并非一件易事，这需要具备专业技能的人进行操作，要对其进行专门的培训。此外，传统台式显微镜的精度有限，其煤岩显微图像的分辨率较低，但高端显微仪器的价格高昂，并不是所以实验室都可以使用先进的仪器获得高清的图片。为此，我们引入了一种新颖的生成对抗网络 (GAN) 来重建低分辨率的显微照片。我们采用WBR模块来避免产生图像伪影并增强非线性拟合能力。此外，在生成器网络中嵌入多尺度注意力模块来捕获多个尺度上的相关性特征。 我们总共使用了470张显微照片作为数据集，实验结果表明，这种方法得到了了31.06 dB的PSNR和0.902的SSIM，显著高于最先进的超分辨率重建方法。
    我们所提出模型的贡献主要有以下三点：
    1.据我们所知，这是第一个专门为提高煤炭显微图片分辨率而设计的深度学习网络模型。实验结果表明这种轻量级超分辨率重建网络可以显着提高图像质量。
    2.
