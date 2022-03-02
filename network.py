from numpy.linalg.linalg import det
from globals import *
import math
logger = Logging().get(__name__, args.loglevel)



class CAN_2d(nn.Module):
    def __init__(self, model_type=PHYS_TYPE):
        super(CAN_2d, self).__init__()

        self.conv1_motion = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2_motion = nn.Conv2d(32, 32, (3, 3))
        self.avgpool1_motion = nn.AvgPool2d((2, 2))
        self.dropout1_motion = nn.Dropout(0.25)
        self.conv3_motion = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv4_motion = nn.Conv2d(64, 64, (3, 3))
        self.avgpool2_motion = nn.AvgPool2d((2, 2))
        self.dropout2_motion = nn.Dropout(0.25)
        if model_type == 'HR':
            self.dense1_motion = nn.Linear(3136, 128)
            self.dropout3_motion = nn.Dropout(0.5)
            self.dense2_motion = nn.Linear(128, 1)
        elif model_type == 'RR':
            self.dense1_motion = nn.Linear(3136, 32)
            self.dropout3_motion = nn.Dropout(0.5)
            self.dense2_motion = nn.Linear(32, 1)
        else:
            logger.error('Unknown phys type')
        # ~~

        self.conv1_appearance = nn.Conv2d(3, 32, (3, 3), padding=1)
        self.conv2_appearance = nn.Conv2d(32, 32, (3, 3))
        self.conv2_attention = nn.Conv2d(32, 1, (1, 1))   # ***

        self.avgpool1_appearance = nn.AvgPool2d((2, 2))
        self.dropout1_appearance = nn.Dropout(0.25)
        self.conv3_appearance = nn.Conv2d(32, 64, (3, 3), padding=1)
        self.conv4_appearance = nn.Conv2d(64, 64, (3, 3))
        self.conv4_attention = nn.Conv2d(64, 1, (1, 1)) # ***     


    def masknorm(self, x):
        xsum = torch.sum(torch.sum(x, axis=2, keepdims=True), axis=3, keepdims=True)
        xshape = x.shape
        ans = (x/xsum)*xshape[2]*xshape[3]*0.5
        return ans


    def forward(self, xm, xa):

        debug = {}
        logger.debug(xm.shape)
        xm = torch.tanh(self.conv1_motion(xm))
        logger.debug(xm.shape)
        xm = torch.tanh(self.conv2_motion(xm))
        logger.debug(xm.shape)

        # ***
        xa = torch.tanh(self.conv1_appearance(xa))
        xa = torch.tanh(self.conv2_appearance(xa))
        ga = self.masknorm(torch.sigmoid(self.conv2_attention(xa)))
        debug['mask1'] = ga

        # ***

        xm = xm * ga
        xm = self.avgpool1_motion(xm)
        logger.debug(xm.shape)
        xm = self.dropout1_motion(xm)
        logger.debug(xm.shape)

        xm = torch.tanh(self.conv3_motion(xm))
        logger.debug(xm.shape)
        xm = torch.tanh(self.conv4_motion(xm))
        logger.debug(xm.shape)
        debug['level1_xm'] = xm

        # ***
        xa = self.avgpool1_appearance(xa)
        xa = self.dropout1_appearance(xa)
        xa = torch.tanh(self.conv3_appearance(xa))
        xa = torch.tanh(self.conv4_appearance(xa))
        ga = self.masknorm(torch.sigmoid(self.conv4_attention(xa)))
        debug['mask2'] = ga
        # ***

        xm = xm * ga
        xm = self.avgpool2_motion(xm)
        logger.debug(xm.shape)
        xm = self.dropout2_motion(xm)
        logger.debug(xm.shape)

        debug['level2_xm'] = xm

        xm = xm.permute(0, 2, 3, 1)
        xm = torch.flatten(xm, 1)
        xm = torch.tanh(self.dense1_motion(xm))
        logger.debug(xm.shape)

        xm = self.dropout3_motion(xm)
        debug['dense1'] = xm
        xm = self.dense2_motion(xm)
        logger.debug(xm.shape)

        
        return xm, xa, debug


    def preprocess(self, x):
        from dataloader import get_appearance_motion
        xa, xm = get_appearance_motion(x)
        return xa, xm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class  Tnet(nn.Module):
    def __init__(self):
        super(Tnet, self).__init__()
        self.batch_size, self.seqlen = args.batch_size, args.seqlen

        ## CAN-2D
        self.can2d = CAN_2d()

        ###
        self.dmodel = 32
        self.feat_win = 4

        self.linear_in = nn.Linear(128 if PHYS_TYPE=='HR' else 32, self.dmodel)
        self.cls_token = nn.Embedding(1, self.dmodel)
        self.pos_encoder = PositionalEncoding(self.dmodel, max_len=args.seqlen+1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.dmodel, nhead=8, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.numlayer)
        
        
        self.linear_out = nn.Sequential(
            nn.Linear(self.dmodel, 1),   
        )

    
    def base_encoder(self, x):
        debug_dict = {}
        xm, xa = x[..., :3], x[...,  3:]
        xm, xa = [_.reshape(-1, 36, 36, 3).permute(0, 3, 1, 2) for _ in [xm, xa]]
        canout, _, can_debug = self.can2d(xm, xa)

        debug_dict['can_xm'] = can_debug['dense1']
        debug_dict['mask1'] = can_debug['mask1']
        debug_dict['canout'] = canout

        return debug_dict


    def forward(self, x, bpsignal):
        dp_debug_dict = {}

        dp_debug_dict = self.base_encoder(x)
        x = dp_debug_dict['can_xm']
        x = self.linear_in(x.reshape(-1, 128 if PHYS_TYPE=='HR' else 32))

        x = x.reshape(-1, self.seqlen , self.dmodel)
        token = self.cls_token.weight.repeat(x.shape[0], 1).unsqueeze(1)
        x = torch.cat([token, x], 1)
        x = x.permute(1,0,2)
        assert x.shape[0] == args.seqlen+1
        
        x = x * np.sqrt(self.dmodel)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1,0,2)

        dp_debug_dict['encoder'] = x
        x = self.linear_out(x.reshape(-1, self.dmodel))
        x = x.reshape(-1, self.seqlen+1)

        x = x[:, 1:]
        assert x.shape[1] == args.seqlen

        return x, dp_debug_dict


