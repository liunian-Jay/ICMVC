data_dict = {
    0: 'Scene-15',
    1: 'LandUse-21',
    2: 'handwritten',
    3: 'MSRC_v1',
    4: 'NoisyMNIST'
}


def get_config(flag=1):
    """Determine the parameter information of the network"""
    data_name = data_dict[flag]
    if data_name in ['Scene-15']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=15,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=128,
            ),
            Autoencoder=dict(
                gcnEncoder1=[40, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[59, 1024, 1024, 1024, 1024 // 2],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            )

        )
    elif data_name in ['LandUse-21']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=21,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=256,
            ),
            Autoencoder=dict(
                gcnEncoder1=[59, 1024, 1024, 1024, 1024 // 4],
                gcnEncoder2=[40, 1024, 1024, 1024, 1024 // 4],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    elif data_name in ['MSRC_v1']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=7,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=256,
            ),
            Autoencoder=dict(
                gcnEncoder1=[576, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[512, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    elif data_name in ['handwritten']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=10,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=256,
            ),
            Autoencoder=dict(
                gcnEncoder1=[76, 1024, 1024, 1024, 1024 // 2],
                gcnEncoder2=[64, 1024, 1024, 1024, 1024 // 2],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
    elif data_name in ['NoisyMNIST']:
        return dict(
            dataset=data_name,
            topk=10,
            missing_rate=0.0,
            n_clusters=10,
            training=dict(
                lr=1.0e-3,
                epoch=500,
                batch_size=256,
            ),
            Autoencoder=dict(
                gcnEncoder1=[784, 1024, 1024, 1024, 1024 // 8],
                gcnEncoder2=[784, 1024, 1024, 1024, 1024 // 8],
                activations1='relu',
                activations2='relu',
                batchnorm=True,
            ),
        )
