"""
Debug: find top-5 selection difference for id_src=20 at step 64.
"""
import kaggle_environments as ke
import math, copy, random
import numpy as np
import pandas as pd
import polars as pl

CENTER = 50.0; SUN_RADIUS = 10.0; ROTATION_RADIUS_LIMIT = 50.0; MAX_SPEED = 6.0; NB_STEPS_SIM = 10; PLANET_MARGIN = 0.1; BOARD_SIZE = 100.0; MAX_NB_STEP = 500

class Obs:
    def __init__(self, planets, initial_planets=None, fleets=None, next_fleet_id=100, comets=None, comet_planet_ids=None, angular_velocity=0.0):
        self.planets = [list(p) for p in planets]
        self.initial_planets = [list(p) for p in (initial_planets if initial_planets is not None else planets)]
        self.fleets = [list(f) for f in (fleets or [])]
        self.next_fleet_id = next_fleet_id; self.comets = comets or []; self.comet_planet_ids = comet_planet_ids or []; self.angular_velocity = angular_velocity

def distance(p1,p2): return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
def point_to_segment_distance(p,v,w):
    l2=(v[0]-w[0])**2+(v[1]-w[1])**2
    if l2==0: return distance(p,v)
    t=max(0,min(1,((p[0]-v[0])*(w[0]-v[0])+(p[1]-v[1])*(w[1]-v[1]))/l2))
    return distance(p,(v[0]+t*(w[0]-v[0]),v[1]+t*(w[1]-v[1])))

def interpreter(obs,actions,step,num_agents=2):
    obs0=obs
    for group in obs0.comets:
        expired=set(pid for i,pid in enumerate(group["planet_ids"]) if group["path_index"]>=len(group["paths"][i]))
        if expired:
            obs0.planets=[p for p in obs0.planets if p[0] not in expired]
            obs0.initial_planets=[p for p in obs0.initial_planets if p[0] not in expired]
            obs0.comet_planet_ids=[pid for pid in obs0.comet_planet_ids if pid not in expired]
            for g in obs0.comets: g["planet_ids"]=[pid for pid in g["planet_ids"] if pid not in expired]
            obs0.comets=[g for g in obs0.comets if g["planet_ids"]]
    def process_moves(pid,action):
        if not action or not isinstance(action,list): return
        for move in action:
            if len(move)!=3: continue
            from_id,angle,ships=move; ships=int(ships)
            fp=next((p for p in obs0.planets if p[0]==from_id),None)
            if fp and fp[1]==pid and fp[5]>=ships and ships>0:
                fp[5]-=ships
                obs0.fleets.append([obs0.next_fleet_id,pid,fp[2]+math.cos(angle)*(fp[4]+0.1),fp[3]+math.sin(angle)*(fp[4]+0.1),angle,from_id,ships])
                obs0.next_fleet_id+=1
    for i in range(num_agents): process_moves(i,actions[i])
    for p in obs0.planets:
        if p[1]!=-1: p[5]+=p[6]
    fr=[]; combat={p[0]:[] for p in obs0.planets}
    for f in obs0.fleets:
        sp=min(1.0+(MAX_SPEED-1.0)*(math.log(f[6])/math.log(1000))**1.5,MAX_SPEED)
        op=(f[2],f[3]); f[2]+=math.cos(f[4])*sp; f[3]+=math.sin(f[4])*sp; np2=(f[2],f[3]); hit=False
        for p in obs0.planets:
            if point_to_segment_distance((p[2],p[3]),op,np2)<p[4]: combat[p[0]].append(f);fr.append(f);hit=True;break
        if hit: continue
        if not(0<=f[2]<=BOARD_SIZE and 0<=f[3]<=BOARD_SIZE): fr.append(f);continue
        if point_to_segment_distance((CENTER,CENTER),op,np2)<SUN_RADIUS: fr.append(f);continue
    cps=set(obs0.comet_planet_ids); ibi={p[0]:p for p in obs0.initial_planets}
    def sw(planet,op,np2):
        if op==np2:return
        for f in obs0.fleets:
            if f not in fr and point_to_segment_distance((f[2],f[3]),op,np2)<planet[4]: combat[planet[0]].append(f);fr.append(f)
    for p in obs0.planets:
        if p[0] in cps: continue
        ip=ibi.get(p[0])
        if not ip: continue
        dx=ip[2]-CENTER;dy=ip[3]-CENTER;r=math.sqrt(dx**2+dy**2);op=(p[2],p[3])
        if r+p[4]<ROTATION_RADIUS_LIMIT: ia=math.atan2(dy,dx);p[2]=CENTER+r*math.cos(ia+obs0.angular_velocity*step);p[3]=CENTER+r*math.sin(ia+obs0.angular_velocity*step)
        sw(p,op,(p[2],p[3]))
    for group in obs0.comets:
        group["path_index"]+=1;idx=group["path_index"]
        for i,pid in enumerate(group["planet_ids"]):
            p=next((x for x in obs0.planets if x[0]==pid),None)
            if not p: continue
            pp=group["paths"][i]
            if idx>=len(pp): pass
            else:
                op=(p[2],p[3]);p[2]=pp[idx][0];p[3]=pp[idx][1]
                if op[0]>=0: sw(p,op,(p[2],p[3]))
    obs0.fleets=[f for f in obs0.fleets if f not in fr]
    for pid,pf in combat.items():
        p=next((x for x in obs0.planets if x[0]==pid),None)
        if not p or not pf: continue
        ps={}
        for f in pf: ps[f[1]]=ps.get(f[1],0)+f[6]
        if not ps: continue
        sp2=sorted(ps.items(),key=lambda x:x[1],reverse=True);tp,ts=sp2[0]
        if len(sp2)>1: ss=ts-sp2[1][1];so=tp if ss>0 else -1
        else: so=tp;ss=ts
        if ss>0:
            if p[1]==so: p[5]+=ss
            else: p[5]-=ss;p[1]=so if p[5]<0 else p[1];p[5]=abs(p[5]) if p[5]<0 else p[5]
    obs0.fleets=[f for f in obs0.fleets if f not in fr]
    return {"planets":obs0.planets,"initial_planets":obs0.initial_planets,"fleets":obs0.fleets,"next_fleet_id":obs0.next_fleet_id,"comets":obs0.comets,"comet_planet_ids":obs0.comet_planet_ids}

def _simulate(obs,global_step,num_agents,n_steps=NB_STEPS_SIM):
    sim=copy.deepcopy(obs);na=[[]for _ in range(num_agents)];rows=[]
    for i in range(n_steps+1):
        for p in sim.planets:
            r=math.hypot(p[2]-CENTER,p[3]-CENTER)
            if p[0] in sim.comet_planet_ids: nat="comet"
            elif r+p[4]<ROTATION_RADIUS_LIMIT: nat="moving"
            else: nat="fix"
            rows.append({"step":global_step+i,"id":p[0],"x":p[2],"y":p[3],"radius":p[4],"ships":p[5],"production":p[6],"owner":p[1],"nature":nat})
        interpreter(sim,na,global_step+i,num_agents)
    return pd.DataFrame(rows)

def random_agent_fn(obs):
    player=obs.player; my=[p for p in obs.planets if p[1]==player]
    if not my: return []
    p=random.choice(my); ships=p[5]//2
    if ships<1: return []
    return [[p[0],random.uniform(0,2*math.pi),ships]]

SEED=42; N_STEPS=100
random.seed(SEED)
env=ke.make("orbit_wars",debug=False)
env.reset(2)

for env_step in range(N_STEPS):
    obs0=env.state[0].observation; obs1=env.state[1].observation
    df=_simulate(copy.deepcopy(obs0),global_step=env_step,num_agents=2,n_steps=NB_STEPS_SIM)

    if env_step == 64:
        print(f"=== Inspecting step {env_step} ===")

        # Build attacks_with_angle (pandas version) for id_src=20
        from collections import namedtuple

        # Get mine_across_sim
        mine_across_sim = (
            df.assign(is_mine=lambda d: (d["owner"]==0).astype(int)).groupby("id")
            .agg(step_src=("step","first"),x_src=("x","first"),y_src=("y","first"),radius_src=("radius","first"),
                 ships_min=("ships","min"),production_src=("production","first"),nature_src=("nature","first"),
                 owner_src=("owner","first"),row_count=("ships","size"),is_mine=("is_mine","sum"))
            .query("row_count == is_mine and owner_src==0")
            .reset_index(drop=False).rename(columns={"id":"id_src"})
        )
        src20 = mine_across_sim[mine_across_sim["id_src"]==20].iloc[0]
        print(f"\nid_src=20: ships_min={src20['ships_min']}, production={src20['production_src']}, step_src={src20['step_src']}")

        # Build expanded_mine for id_src=20
        expanded_mine = (
            mine_across_sim
            .assign(ships_sent=(mine_across_sim["ships_min"]+mine_across_sim["production_src"]*NB_STEPS_SIM).apply(lambda n: list(range(1,n+1))))
            .explode("ships_sent").astype({"ships_sent":int}).reset_index(drop=True)
        )

        # Cross join with df
        df_src_tgt = (
            expanded_mine[expanded_mine["id_src"]==20]
            .merge(df[df["id"]==32], how="cross")
            .query("step > step_src and id != id_src")
        )

        # Compute possible attacks for id_src=20 -> id=32
        pa = (
            df_src_tgt.assign(
                dist_tgt_src=lambda d: ((d["x"]-d["x_src"])**2+(d["y"]-d["y_src"])**2)**0.5,
                step_diff=lambda d: d["step"]-d["step_src"],
                fleet_speed=lambda d: 1.0+(MAX_SPEED-1.0)*(np.log(d["ships_sent"])/math.log(1000))**1.5,
                dist_fleet_src_min=lambda d: d["step_diff"]*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
                dist_fleet_src_max=lambda d: (d["step_diff"]+1)*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
                collision=lambda d: (
                    ((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_min"])&(d["dist_fleet_src_min"]<d["dist_tgt_src"]+d["radius"]))|
                    ((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_max"])&(d["dist_fleet_src_max"]<d["dist_tgt_src"]+d["radius"]))
                ))
            .query("collision")
        )

        # Filter ships_sent=18 step=71
        row_18_71 = pa[(pa["ships_sent"]==18) & (pa["step"]==71)]
        print(f"\nPD: id_src=20 -> id=32, ships_sent=18, step=71: {len(row_18_71)} rows")
        if not row_18_71.empty:
            print(row_18_71[["ships_sent","step","ships_min","dist_tgt_src","dist_fleet_src_min","dist_fleet_src_max","collision"]].to_string())

        # Now what does the TOP-5 look like for id_src=20?
        # First we need all attacks_with_angle, build the whole pipeline for id_src=20
        # But let's just check: what targets are in top-5 for id_src=20 in pandas?

        # Build full possible_attacks for just id_src=20
        def pt_seg_dist(p, v, w):
            l2=(v[0]-w[0])**2+(v[1]-w[1])**2
            if l2==0: return distance(p,v)
            t=max(0,min(1,((p[0]-v[0])*(w[0]-v[0])+(p[1]-v[1])*(w[1]-v[1]))/l2))
            return distance(p,(v[0]+t*(w[0]-v[0]),v[1]+t*(w[1]-v[1])))

        df_st_20 = expanded_mine[expanded_mine["id_src"]==20].merge(df,how="cross").query("step > step_src and id != id_src")
        pa_20 = (
            df_st_20.assign(
                dist_tgt_src=lambda d:((d["x"]-d["x_src"])**2+(d["y"]-d["y_src"])**2)**0.5,
                step_diff=lambda d:d["step"]-d["step_src"],
                fleet_speed=lambda d:1.0+(MAX_SPEED-1.0)*(np.log(d["ships_sent"])/math.log(1000))**1.5,
                dist_fleet_src_min=lambda d:d["step_diff"]*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
                dist_fleet_src_max=lambda d:(d["step_diff"]+1)*d["fleet_speed"]+PLANET_MARGIN+d["radius_src"],
                collision=lambda d:((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_min"])&(d["dist_fleet_src_min"]<d["dist_tgt_src"]+d["radius"]))|((d["dist_tgt_src"]-d["radius"]<d["dist_fleet_src_max"])&(d["dist_fleet_src_max"]<d["dist_tgt_src"]+d["radius"])))
            .query("collision")
            .assign(crossing_sun=lambda d:d.apply(lambda row:pt_seg_dist((CENTER,CENTER),(row["x_src"],row["y_src"]),(row["x"],row["y"]))<SUN_RADIUS+PLANET_MARGIN,axis=1).astype(bool))
            .query("not crossing_sun")
            .assign(
                angle=lambda d:np.arctan2(d["y"]-d["y_src"],d["x"]-d["x_src"]),
                radius_angle=lambda d:np.maximum(
                    np.arccos(((d["dist_tgt_src"]**2+d["dist_fleet_src_min"]**2-d["radius"]**2)/(2*d["dist_tgt_src"]*d["dist_fleet_src_min"])).clip(-1,1)),
                    np.arccos(((d["dist_tgt_src"]**2+d["dist_fleet_src_max"]**2-d["radius"]**2)/(2*d["dist_tgt_src"]*d["dist_fleet_src_max"])).clip(-1,1))
                ),
                angle_min=lambda d:np.mod(d["angle"]-d["radius_angle"],2*math.pi),
                angle_max=lambda d:np.mod(d["angle"]+d["radius_angle"],2*math.pi),
            ).sort_values("step",ascending=True)
        )
        print(f"\nPA_20 rows: {len(pa_20)}, unique id targets: {sorted(pa_20['id'].unique())}")

        # Top-5 for id_src=20 in pandas style
        top5_pd = (
            pa_20.sort_values(["step","ships_sent"],ascending=True)
            .groupby(["id_src","id"],as_index=False).first()
            .sort_values(["step","ships_sent"],ascending=True)
        )
        print(f"\nTop-k sorted by (step,ships_sent) for id_src=20 (all unique targets):")
        print(top5_pd[["id_src","id","ships_sent","step"]].to_string())
        print(f"\nPandas groupby head(5) picks:")
        print(top5_pd.head(5)[["id_src","id","ships_sent","step"]].to_string())

        break

    rng_action=random_agent_fn(obs1)
    env.step([moves:=[[]], rng_action])
    if env.state[0].status!="ACTIVE": break
