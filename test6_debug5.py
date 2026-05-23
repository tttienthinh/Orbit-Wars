"""
Focused debug: find why pandas top-5 for id_src=20 does not include id=32.
"""
import kaggle_environments as ke
import math, copy, random
import numpy as np
import pandas as pd
import polars as pl

CENTER = 50.0; SUN_RADIUS = 10.0; ROTATION_RADIUS_LIMIT = 50.0; MAX_SPEED = 6.0; NB_STEPS_SIM = 10; PLANET_MARGIN = 0.1; BOARD_SIZE = 100.0; MAX_NB_STEP = 500

class Obs:
    def __init__(self, planets, initial_planets=None, fleets=None, next_fleet_id=100, comets=None, comet_planet_ids=None, angular_velocity=0.0):
        self.planets=[list(p) for p in planets]; self.initial_planets=[list(p) for p in (initial_planets if initial_planets is not None else planets)]
        self.fleets=[list(f) for f in (fleets or [])]; self.next_fleet_id=next_fleet_id; self.comets=comets or []; self.comet_planet_ids=comet_planet_ids or []; self.angular_velocity=angular_velocity

def distance(p1,p2): return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
def point_to_segment_distance(p,v,w):
    l2=(v[0]-w[0])**2+(v[1]-w[1])**2
    if l2==0:return distance(p,v)
    t=max(0,min(1,((p[0]-v[0])*(w[0]-v[0])+(p[1]-v[1])*(w[1]-v[1]))/l2))
    return distance(p,(v[0]+t*(w[0]-v[0]),v[1]+t*(w[1]-v[1])))

def interpreter(obs,actions,step,num_agents=2):
    obs0=obs
    expired_comet_pids=[]
    for group in obs0.comets:
        idx=group["path_index"]
        for i,pid in enumerate(group["planet_ids"]):
            if idx>=len(group["paths"][i]):expired_comet_pids.append(pid)
    if expired_comet_pids:
        expired_set=set(expired_comet_pids);obs0.planets=[p for p in obs0.planets if p[0] not in expired_set];obs0.initial_planets=[p for p in obs0.initial_planets if p[0] not in expired_set];obs0.comet_planet_ids=[pid for pid in obs0.comet_planet_ids if pid not in expired_set]
        for g in obs0.comets:g["planet_ids"]=[pid for pid in g["planet_ids"] if pid not in expired_set]
        obs0.comets=[g for g in obs0.comets if g["planet_ids"]]
    def process_moves(pid,action):
        if not action or not isinstance(action,list):return
        for move in action:
            if len(move)!=3:continue
            from_id,angle,ships=move;ships=int(ships);fp=next((p for p in obs0.planets if p[0]==from_id),None)
            if fp and fp[1]==pid and fp[5]>=ships and ships>0:fp[5]-=ships;obs0.fleets.append([obs0.next_fleet_id,pid,fp[2]+math.cos(angle)*(fp[4]+0.1),fp[3]+math.sin(angle)*(fp[4]+0.1),angle,from_id,ships]);obs0.next_fleet_id+=1
    for i in range(num_agents):process_moves(i,actions[i])
    for p in obs0.planets:
        if p[1]!=-1:p[5]+=p[6]
    fr=[];combat={p[0]:[] for p in obs0.planets}
    for f in obs0.fleets:
        sp=min(1.0+(MAX_SPEED-1.0)*(math.log(f[6])/math.log(1000))**1.5,MAX_SPEED);op=(f[2],f[3]);f[2]+=math.cos(f[4])*sp;f[3]+=math.sin(f[4])*sp;np2=(f[2],f[3]);hit=False
        for p in obs0.planets:
            if point_to_segment_distance((p[2],p[3]),op,np2)<p[4]:combat[p[0]].append(f);fr.append(f);hit=True;break
        if hit:continue
        if not(0<=f[2]<=BOARD_SIZE and 0<=f[3]<=BOARD_SIZE):fr.append(f);continue
        if point_to_segment_distance((CENTER,CENTER),op,np2)<SUN_RADIUS:fr.append(f);continue
    cps=set(obs0.comet_planet_ids);ibi={p[0]:p for p in obs0.initial_planets}
    def sw(planet,op,np2):
        if op==np2:return
        for f in obs0.fleets:
            if f not in fr and point_to_segment_distance((f[2],f[3]),op,np2)<planet[4]:combat[planet[0]].append(f);fr.append(f)
    for p in obs0.planets:
        if p[0] in cps:continue
        ip=ibi.get(p[0])
        if not ip:continue
        dx=ip[2]-CENTER;dy=ip[3]-CENTER;r=math.sqrt(dx**2+dy**2);op=(p[2],p[3])
        if r+p[4]<ROTATION_RADIUS_LIMIT:ia=math.atan2(dy,dx);p[2]=CENTER+r*math.cos(ia+obs0.angular_velocity*step);p[3]=CENTER+r*math.sin(ia+obs0.angular_velocity*step)
        sw(p,op,(p[2],p[3]))
    expired_comet_pids2=[]
    for group in obs0.comets:
        group["path_index"]+=1;idx=group["path_index"]
        for i,pid in enumerate(group["planet_ids"]):
            p=next((x for x in obs0.planets if x[0]==pid),None)
            if not p:continue
            pp=group["paths"][i]
            if idx>=len(pp):expired_comet_pids2.append(pid)
            else:op=(p[2],p[3]);p[2]=pp[idx][0];p[3]=pp[idx][1];(op[0]>=0) and sw(p,op,(p[2],p[3]))
    if expired_comet_pids2:
        es=set(expired_comet_pids2);obs0.planets=[p for p in obs0.planets if p[0] not in es];obs0.initial_planets=[p for p in obs0.initial_planets if p[0] not in es];obs0.comet_planet_ids=[pid for pid in obs0.comet_planet_ids if pid not in es]
        for g in obs0.comets:g["planet_ids"]=[pid for pid in g["planet_ids"] if pid not in es]
        obs0.comets=[g for g in obs0.comets if g["planet_ids"]]
    obs0.fleets=[f for f in obs0.fleets if f not in fr]
    for pid,pf in combat.items():
        p=next((x for x in obs0.planets if x[0]==pid),None)
        if not p or not pf:continue
        ps={}
        for f in pf:ps[f[1]]=ps.get(f[1],0)+f[6]
        if not ps:continue
        sp2=sorted(ps.items(),key=lambda x:x[1],reverse=True);tp,ts=sp2[0]
        if len(sp2)>1:ss=ts-sp2[1][1];so=(tp if ss>0 else -1)
        else:so=tp;ss=ts
        if ss>0:
            if p[1]==so:p[5]+=ss
            else:p[5]-=ss;(p[5]<0) and setattr(p,'__class__',type(p)) or None
            if p[5]<0:p[1]=so;p[5]=abs(p[5])
    obs0.fleets=[f for f in obs0.fleets if f not in fr]
    return {"planets":obs0.planets,"initial_planets":obs0.initial_planets,"fleets":obs0.fleets,"next_fleet_id":obs0.next_fleet_id,"comets":obs0.comets,"comet_planet_ids":obs0.comet_planet_ids}

def _simulate(obs,global_step,num_agents,n_steps=NB_STEPS_SIM):
    sim=copy.deepcopy(obs);na=[[]for _ in range(num_agents)];rows=[]
    for i in range(n_steps+1):
        for p in sim.planets:
            r=math.hypot(p[2]-CENTER,p[3]-CENTER)
            nat="comet" if p[0] in sim.comet_planet_ids else ("moving" if r+p[4]<ROTATION_RADIUS_LIMIT else "fix")
            rows.append({"step":global_step+i,"id":p[0],"x":p[2],"y":p[3],"radius":p[4],"ships":p[5],"production":p[6],"owner":p[1],"nature":nat})
        interpreter(sim,na,global_step+i,num_agents)
    return pd.DataFrame(rows)

def random_agent_fn(obs):
    player=obs.player;my=[p for p in obs.planets if p[1]==player]
    if not my:return []
    p=random.choice(my);ships=p[5]//2
    if ships<1:return []
    return [[p[0],random.uniform(0,2*math.pi),ships]]

SEED=42;N_STEPS=100
random.seed(SEED)
env=ke.make("orbit_wars",debug=False)
env.reset(2)

prev_moves_pd = []

for env_step in range(N_STEPS):
    obs0=env.state[0].observation;obs1=env.state[1].observation
    df=_simulate(copy.deepcopy(obs0),global_step=env_step,num_agents=2,n_steps=NB_STEPS_SIM)

    if env_step == 64:
        print(f"=== Inspecting step {env_step} ===")
        # Build mine_across_sim
        mine = (
            df.assign(is_mine=lambda d:(d["owner"]==0).astype(int)).groupby("id")
            .agg(step_src=("step","first"),x_src=("x","first"),y_src=("y","first"),radius_src=("radius","first"),
                 ships_min=("ships","min"),production_src=("production","first"),nature_src=("nature","first"),
                 owner_src=("owner","first"),row_count=("ships","size"),is_mine=("is_mine","sum"))
            .query("row_count == is_mine and owner_src==0")
            .reset_index(drop=False).rename(columns={"id":"id_src"})
        )
        src20=mine[mine["id_src"]==20].iloc[0]
        print(f"id_src=20: ships_min={src20['ships_min']}, production={src20['production_src']}, step_src={src20['step_src']}")

        # Build full possible_attacks for id_src=20
        mine20=mine[mine["id_src"]==20].copy()
        mine20["ships_sent"]=mine20.apply(lambda r: list(range(1,int(r["ships_min"]+r["production_src"]*NB_STEPS_SIM)+1)),axis=1)
        mine20=mine20.explode("ships_sent").astype({"ships_sent":int}).reset_index(drop=True)
        df_st=mine20.merge(df,how="cross").query("step > step_src and id != id_src")
        pa=(df_st
            .assign(dist_tgt_src=lambda d:((d["x"]-d["x_src"])**2+(d["y"]-d["y_src"])**2)**0.5,
                    step_diff=lambda d:d["step"]-d["step_src"],
                    fleet_speed=lambda d:1.0+(MAX_SPEED-1.0)*(np.log(d["ships_sent"])/math.log(1000))**1.5,
                    dist_fleet_src_min=lambda d:d["step_diff"]*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
                    dist_fleet_src_max=lambda d:(d["step_diff"]+1)*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
                    collision=lambda d:((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_min"])&(d["dist_fleet_src_min"]<d["dist_tgt_src"]+d["radius"]))|((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_max"])&(d["dist_fleet_src_max"]<d["dist_tgt_src"]+d["radius"])))
            .query("collision")
            .assign(crossing_sun=lambda d:d.apply(lambda row:point_to_segment_distance((CENTER,CENTER),(row["x_src"],row["y_src"]),(row["x"],row["y"]))<SUN_RADIUS+PLANET_MARGIN,axis=1).astype(bool))
            .query("not crossing_sun")
            .assign(angle=lambda d:np.arctan2(d["y"]-d["y_src"],d["x"]-d["x_src"]),
                    radius_angle=lambda d:np.maximum(
                        np.arccos(((d["dist_tgt_src"]**2+d["dist_fleet_src_min"]**2-d["radius"]**2)/(2*d["dist_tgt_src"]*d["dist_fleet_src_min"])).clip(-1,1)),
                        np.arccos(((d["dist_tgt_src"]**2+d["dist_fleet_src_max"]**2-d["radius"]**2)/(2*d["dist_tgt_src"]*d["dist_fleet_src_max"])).clip(-1,1))),
                    angle_min=lambda d:np.mod(d["angle"]-d["radius_angle"],2*math.pi),
                    angle_max=lambda d:np.mod(d["angle"]+d["radius_angle"],2*math.pi))
            .sort_values("step",ascending=True)
        )
        print(f"id_src=20 possible_attacks: {len(pa)} rows, targets: {sorted(pa['id'].unique())}")

        # Top-5 in pandas style:
        # Step 1: groupby ["id_src","id"], take first sorted by [step,ships_sent]
        top_per_pair = (
            pa.sort_values(["step","ships_sent"],ascending=True)
            .groupby(["id_src","id"],as_index=False).first()
            .sort_values(["step","ships_sent"],ascending=True)
        )
        print(f"\nTop-per-pair sorted (all unique targets for id_src=20):")
        print(top_per_pair[["id_src","id","step","ships_sent"]].to_string())

        # Step 2: head(5) for id_src=20
        top5 = top_per_pair.groupby("id_src",as_index=False).head(5)
        print(f"\nTop-5 for id_src=20 (pandas):")
        print(top5[["id_src","id","step","ships_sent"]].to_string())

        # Polars version
        df_lf = pl.from_pandas(df).sort("step").lazy()
        mine_pl = (
            df_lf.with_columns(pl.when(pl.col("owner")==0).then(1).otherwise(0).alias("is_mine"))
            .group_by("id",maintain_order=True)
            .agg(pl.first("step").alias("step_src"),pl.first("x").alias("x_src"),pl.first("y").alias("y_src"),pl.first("radius").alias("radius_src"),pl.min("ships").alias("ships_min"),pl.first("production").alias("production_src"),pl.first("nature").alias("nature_src"),pl.first("owner").alias("owner_src"),pl.len().alias("row_count"),pl.sum("is_mine").alias("is_mine"))
            .filter((pl.col("row_count")==pl.col("is_mine"))&(pl.col("owner_src")==0))
            .rename({"id":"id_src"}).collect()
        )
        mine20_pl = mine_pl.filter(pl.col("id_src")==20)
        dx_vw=pl.col("x")-pl.col("x_src");dy_vw=pl.col("y")-pl.col("y_src");l2=dx_vw.pow(2)+dy_vw.pow(2)
        dot=(CENTER-pl.col("x_src"))*dx_vw+(CENTER-pl.col("y_src"))*dy_vw
        t=(dot/pl.when(l2==0).then(pl.lit(1.0)).otherwise(l2)).clip(0.0,1.0)
        dsp=((CENTER-(pl.col("x_src")+t*dx_vw)).pow(2)+(CENTER-(pl.col("y_src")+t*dy_vw)).pow(2)).sqrt()
        dsd=((CENTER-pl.col("x_src")).pow(2)+(CENTER-pl.col("y_src")).pow(2)).sqrt()
        dts=pl.when(l2==0).then(dsd).otherwise(dsp)
        cs_expr=dts<(SUN_RADIUS+PLANET_MARGIN)
        dte=((pl.col("x")-pl.col("x_src")).pow(2)+(pl.col("y")-pl.col("y_src")).pow(2)).sqrt()
        sde=pl.col("step")-pl.col("step_src")
        fse=1.0+(MAX_SPEED-1.0)*(pl.col("ships_sent").cast(pl.Float64).log(base=math.e)/math.log(1000.0)).pow(1.5)
        dmine=sde*fse+PLANET_MARGIN+pl.col("radius_src");dmaxe=(sde+1)*fse+PLANET_MARGIN+pl.col("radius_src")
        cole=((dte-pl.col("radius")<dmine)&(dmine<dte+pl.col("radius")))|((dte-pl.col("radius")<dmaxe)&(dmaxe<dte+pl.col("radius")))
        pa_pl=(mine20_pl.lazy()
            .with_columns(pl.int_ranges(1,pl.col("ships_min")+pl.col("production_src")*NB_STEPS_SIM+1,dtype=pl.Int64).alias("ships_sent"))
            .explode("ships_sent").join(df_lf,how="cross")
            .filter((pl.col("step")>pl.col("step_src"))&(pl.col("id")!=pl.col("id_src")))
            .with_columns([dte.alias("dist_tgt_src"),sde.alias("step_diff"),fse.alias("fleet_speed"),dmine.alias("dist_fleet_src_min"),dmaxe.alias("dist_fleet_src_max"),cole.alias("collision")])
            .filter(pl.col("collision")).with_columns(cs_expr.alias("crossing_sun")).filter(~pl.col("crossing_sun"))
            .with_columns(pl.arctan2(pl.col("y")-pl.col("y_src"),pl.col("x")-pl.col("x_src")).alias("angle"))
            .with_columns(pl.max_horizontal(
                ((pl.col("dist_tgt_src").pow(2)+pl.col("dist_fleet_src_min").pow(2)-pl.col("radius").pow(2))/(2*pl.col("dist_tgt_src")*pl.col("dist_fleet_src_min"))).clip(-1.0,1.0).arccos(),
                ((pl.col("dist_tgt_src").pow(2)+pl.col("dist_fleet_src_max").pow(2)-pl.col("radius").pow(2))/(2*pl.col("dist_tgt_src")*pl.col("dist_fleet_src_max"))).clip(-1.0,1.0).arccos(),
            ).alias("radius_angle"))
            .with_columns([((pl.col("angle")-pl.col("radius_angle"))%(2*math.pi)).alias("angle_min"),((pl.col("angle")+pl.col("radius_angle"))%(2*math.pi)).alias("angle_max")])
            .sort("step").collect()
        )
        print(f"\nid_src=20 polars possible_attacks: {len(pa_pl)} rows, targets: {sorted(pa_pl['id'].to_list())[:10]}...")

        # Polars top-per-pair
        top_per_pair_pl=(
            pa_pl.lazy().sort(["step","ships_sent"]).group_by(["id_src","id"],maintain_order=True).first()
            .sort(["step","ships_sent"]).collect()
        )
        print(f"\nTop-per-pair sorted (polars):")
        print(top_per_pair_pl.select(["id_src","id","step","ships_sent"]).to_pandas().to_string())

        top5_pl = top_per_pair_pl.lazy().group_by("id_src",maintain_order=True).head(5).collect()
        print(f"\nTop-5 for id_src=20 (polars):")
        print(top5_pl.select(["id_src","id","step","ships_sent"]).to_pandas().to_string())

        break

    rng_action=random_agent_fn(obs1)
    # We need to submit moves_pd for player 0 - just submit empty to advance game consistently
    env.step([[], rng_action])
    if env.state[0].status!="ACTIVE":
        print(f"Game ended at step {env_step}")
        break

print("Done")
